"""
PharmAI Intelligence Dashboard — Flask Backend  v3.0
- Goal 2: K-Means uses silhouette-optimized k (matches notebook exactly)
- All endpoints expose full graph data: elbow, silhouette, residuals, learning curves
- Goal 4: Ridge + XGBoost prediction scatter + residual plots
- Advanced A: SVD variance curve + CF score distribution
- Advanced B: Isolation Forest score histogram + AE reconstruction error curve
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings, traceback

warnings.filterwarnings('ignore')

# ── DB layer ──────────────────────────────────────────────────────────────────
from db_connection import (
    get_connection, run_query,
    QUERY_GOAL1_TIMESERIES,
    QUERY_GOAL2_SEGMENTATION,
    QUERY_GOAL3_CREDIT,
    QUERY_GOAL4_MARGIN,
)

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression, Ridge
from sklearn.ensemble         import RandomForestClassifier, IsolationForest
from sklearn.cluster          import KMeans, DBSCAN
from sklearn.decomposition    import TruncatedSVD, PCA
from sklearn.metrics          import (
    silhouette_score, davies_bouldin_score,
    confusion_matrix, roc_auc_score, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection  import train_test_split, learning_curve
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
CORS(app)

# ── Shared live state (read by chatbot) ───────────────────────────────────────
DASHBOARD_STATE = {
    "goal1": {}, "goal2": {}, "goal3": {},
    "goal4": {}, "advA":  {}, "advB":  {}, "kpis": {},
}

def db():
    return get_connection()


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL 1 — Sales Forecasting  (SARIMA + XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/goal1/timeseries', methods=['GET'])
def goal1_timeseries():
    try:
        c  = db()
        df = run_query(QUERY_GOAL1_TIMESERIES, c); c.close()

        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df = df.sort_values('sale_date').reset_index(drop=True)

        dfw = (df.groupby(pd.Grouper(key='sale_date', freq='W'))
                 .agg(total_revenue=('total_revenue','sum'),
                      total_qty_sold=('total_qty_sold','sum'),
                      nb_customers=('nb_customers','sum'))
                 .reset_index())
        dfw = dfw[dfw['total_revenue'] > 0].reset_index(drop=True)
        dfw['week']    = dfw['sale_date'].dt.isocalendar().week.astype(int)
        dfw['month']   = dfw['sale_date'].dt.month
        dfw['quarter'] = dfw['sale_date'].dt.quarter

        n, train_n = len(dfw), int(len(dfw) * 0.75)
        train, test = dfw.iloc[:train_n], dfw.iloc[train_n:]

        # SARIMA
        try:
            sarima     = SARIMAX(train['total_revenue'], order=(1,1,1),
                                  seasonal_order=(1,1,0,52),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
            sarima_pred = sarima.forecast(steps=len(test)).tolist()
            sarima_fut  = sarima.forecast(steps=4).tolist()
        except Exception:
            sarima_pred = test['total_revenue'].tolist()
            sarima_fut  = [float(dfw['total_revenue'].iloc[-1])] * 4

        # XGBoost
        fcols = ['week','month','quarter']
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(dfw.iloc[:train_n][fcols], dfw.iloc[:train_n]['total_revenue'])
        xgb_pred = xgb.predict(test[fcols]).tolist()

        last_date   = dfw['sale_date'].iloc[-1]
        fut_dates   = [last_date + timedelta(weeks=i+1) for i in range(4)]
        fut_X       = pd.DataFrame({'week':[d.isocalendar()[1] for d in fut_dates],
                                    'month':[d.month for d in fut_dates],
                                    'quarter':[d.quarter for d in fut_dates]})
        xgb_fut     = xgb.predict(fut_X).tolist()

        act  = test['total_revenue'].tolist()
        mape = lambda a,p: round(float(np.mean(np.abs((np.array(a)-np.array(p))/np.maximum(np.array(a),1)))*100),2)
        rmse = lambda a,p: round(float(np.sqrt(mean_squared_error(a,p))))
        mae  = lambda a,p: round(float(mean_absolute_error(a,p)))

        metrics = {
            "sarima":  {"MAPE":mape(act,sarima_pred),"RMSE":rmse(act,sarima_pred),"MAE":mae(act,sarima_pred)},
            "xgboost": {"MAPE":mape(act,xgb_pred),  "RMSE":rmse(act,xgb_pred),  "MAE":mae(act,xgb_pred)},
        }

        # Decomposition
        trend    = dfw['total_revenue'].rolling(4,center=True).mean().bfill().ffill().tolist()
        seasonal = (dfw['total_revenue'] - pd.Series(trend)).tolist()
        residual = (dfw['total_revenue'] - pd.Series(trend) - pd.Series(seasonal)).tolist()

        # ── XGBoost learning curve (train size vs RMSE) ──
        train_sizes, train_scores, val_scores = learning_curve(
            XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            dfw[fcols], dfw['total_revenue'],
            train_sizes=np.linspace(0.2, 1.0, 6),
            scoring='neg_root_mean_squared_error', cv=3
        )
        lc_data = {
            "train_sizes": train_sizes.tolist(),
            "train_rmse":  (-train_scores.mean(axis=1)).round(2).tolist(),
            "val_rmse":    (-val_scores.mean(axis=1)).round(2).tolist(),
        }

        # ── Residual plot data (XGBoost) ──
        xgb_all_pred = xgb.predict(dfw[fcols]).tolist()
        residuals_xgb = (dfw['total_revenue'] - pd.Series(xgb_all_pred)).round(2).tolist()

        DASHBOARD_STATE['goal1'] = {
            "weeks_of_data":      n,
            "avg_weekly_revenue": round(float(dfw['total_revenue'].mean()),2),
            "last_revenue":       round(float(dfw['total_revenue'].iloc[-1]),2),
            "sarima_mape":        metrics['sarima']['MAPE'],
            "xgb_mape":           metrics['xgboost']['MAPE'],
            "forecast_4w":        [round(v,2) for v in xgb_fut],
            "best_model":         "XGBoost" if metrics['xgboost']['MAPE']<metrics['sarima']['MAPE'] else "SARIMA",
        }

        return jsonify({
            "train":   {"dates":dfw.iloc[:train_n]['sale_date'].dt.strftime('%Y-%m-%d').tolist(),
                        "revenue":dfw.iloc[:train_n]['total_revenue'].round(2).tolist(),
                        "qty":dfw.iloc[:train_n]['total_qty_sold'].tolist()},
            "test":    {"dates":test['sale_date'].dt.strftime('%Y-%m-%d').tolist(),
                        "actual_revenue":[round(v,2) for v in act],
                        "sarima_revenue":[round(v,2) for v in sarima_pred],
                        "xgb_revenue":   [round(v,2) for v in xgb_pred]},
            "forecast":{"dates":[d.strftime('%Y-%m-%d') for d in fut_dates],
                        "revenue":[round(v,2) for v in xgb_fut],
                        "sarima": [round(v,2) for v in sarima_fut]},
            "metrics": metrics,
            "decomposition":{"dates":dfw['sale_date'].dt.strftime('%Y-%m-%d').tolist(),
                             "trend":[round(v,2) for v in trend],
                             "seasonal":[round(v,2) for v in seasonal],
                             "residual":[round(v,2) for v in residual]},
            # ── NEW graph data ──
            "learning_curve": lc_data,
            "residuals": {
                "dates":    dfw['sale_date'].dt.strftime('%Y-%m-%d').tolist(),
                "actual":   dfw['total_revenue'].round(2).tolist(),
                "predicted":xgb_all_pred,
                "residuals":residuals_xgb,
            },
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL 2 — Customer Segmentation  (K-Means silhouette-optimized + DBSCAN)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/goal2/segmentation', methods=['GET'])
def goal2_segmentation():
    try:
        c  = db()
        df = run_query(QUERY_GOAL2_SEGMENTATION, c); c.close()

        rfm    = df[['recency_days','nb_transactions','total_revenue']].fillna(0)
        scaler = StandardScaler()
        X      = scaler.fit_transform(rfm)

        # ── Elbow + Silhouette sweep (k=2..8) — exactly like notebook ──
        K_range   = range(2, 9)
        inertias  = []
        sil_scores = []
        for k in K_range:
            km_tmp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = km_tmp.fit_predict(X)
            inertias.append(round(float(km_tmp.inertia_), 4))
            sil_scores.append(round(float(silhouette_score(X, labels)), 4))

        best_k   = list(K_range)[int(np.argmax(sil_scores))]
        best_sil = max(sil_scores)

        elbow_data = {
            "k":         list(K_range),
            "inertia":   inertias,
            "silhouette":sil_scores,
            "best_k":    best_k,
            "best_sil":  round(best_sil, 3),
        }

        # ── Fit best K-Means ──
        km     = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
        df['km_cluster'] = km.fit_predict(X)
        km_db  = round(float(davies_bouldin_score(X, df['km_cluster'])), 3)

        # Label clusters by avg revenue rank
        rev_rank = (df.groupby('km_cluster')['total_revenue']
                      .mean().sort_values(ascending=False).index.tolist())
        segment_names = ['Champions','Loyal','At-Risk','Dormant','Fringe','Inactive','Micro']
        lmap = {rev_rank[i]: segment_names[i] for i in range(best_k)}
        df['segment'] = df['km_cluster'].map(lmap)

        # ── DBSCAN ──
        db_m = DBSCAN(eps=0.7, min_samples=5)
        df['db_cluster'] = db_m.fit_predict(X)
        db_noise = int((df['db_cluster'] == -1).sum())
        db_ncl   = int(df['db_cluster'].nunique() - (1 if -1 in df['db_cluster'].values else 0))
        valid_db = df[df['db_cluster'] != -1]
        _db_n_unique = valid_db['db_cluster'].nunique()
        db_sil = (
            round(float(silhouette_score(X[valid_db.index], valid_db['db_cluster'])), 3)
            if len(valid_db) > 1 and _db_n_unique >= 2
            else 0
        )

        # ── PCA 2D ──
        coords = PCA(n_components=2).fit_transform(X)
        df['pca_x'] = coords[:,0].round(4)
        df['pca_y'] = coords[:,1].round(4)

        # ── Silhouette sample scores (for silhouette plot) ──
        from sklearn.metrics import silhouette_samples
        sil_vals = silhouette_samples(X, df['km_cluster'])
        sil_plot = []
        for cl in sorted(df['km_cluster'].unique()):
            vals = sil_vals[df['km_cluster'] == cl].tolist()
            sil_plot.append({
                "cluster": int(cl),
                "segment": lmap[cl],
                "values":  [round(v,4) for v in sorted(vals, reverse=True)],
                "mean":    round(float(np.mean(vals)),4),
            })

        segments = list(lmap.values())
        seg_counts = df['segment'].value_counts().to_dict()
        cluster_profiles = {}
        for s in segments:
            sub = df[df['segment']==s]
            if len(sub) == 0: continue
            cluster_profiles[s] = {
                "avg_recency":   round(float(sub['recency_days'].mean()),1),
                "avg_frequency": round(float(sub['nb_transactions'].mean()),1),
                "avg_monetary":  round(float(sub['total_revenue'].mean()),0),
                "count":         int(len(sub)),
            }

        customers_out = df[['ClientID','segment','recency_days','nb_transactions',
                             'total_revenue','pca_x','pca_y']].rename(
            columns={'ClientID':'id','recency_days':'recency',
                     'nb_transactions':'frequency','total_revenue':'monetary'}
        ).to_dict(orient='records')

        DASHBOARD_STATE['goal2'] = {
            "total_customers": len(df),
            "best_k":          best_k,
            "best_sil":        round(best_sil,3),
            "segment_counts":  seg_counts,
            "km_silhouette":   round(best_sil,3),
            "db_n_clusters":   db_ncl,
        }

        return jsonify({
            "customers":            customers_out,
            "segment_distribution": seg_counts,
            "cluster_profiles":     cluster_profiles,
            "metrics": {
                "kmeans": {"silhouette":round(best_sil,3),"davies_bouldin":km_db,"k":best_k},
                "dbscan": {"silhouette":db_sil,"n_clusters":db_ncl,
                           "noise_pct":round(db_noise/len(df)*100,1)},
            },
            # ── NEW graph data ──
            "elbow_data":     elbow_data,
            "silhouette_plot":sil_plot,
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL 3 — Creditworthiness  (Logistic Regression + Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/goal3/credit', methods=['GET'])
def goal3_credit():
    try:
        c  = db()
        df = run_query(QUERY_GOAL3_CREDIT, c); c.close()

        fcols = [
            'nb_invoices','total_invoiced','total_receivable','total_sales',
            'avg_payment_delay','max_payment_delay','nb_paid','nb_unpaid',
            'payment_rate','avg_discount_pct','total_discount',
            'avg_invoice_value','max_invoice_value','total_returns','nb_products_bought',
        ]
        df  = df.dropna(subset=fcols+['computed_risk'])
        X   = df[fcols].fillna(0)
        le  = LabelEncoder()
        y   = le.fit_transform(df['computed_risk'])
        classes = le.classes_.tolist()

        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        # LR
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_tr)
        lr_pred  = lr.predict(X_te_s)
        lr_proba = lr.predict_proba(X_te_s)
        lr_acc   = round(float((lr_pred==y_te).mean()),3)
        lr_auc   = round(float(roc_auc_score(y_te,lr_proba,multi_class='ovr')),3)
        lr_cm    = confusion_matrix(y_te,lr_pred).tolist()

        # RF
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_pred  = rf.predict(X_te)
        rf_proba = rf.predict_proba(X_te)
        rf_acc   = round(float((rf_pred==y_te).mean()),3)
        rf_auc   = round(float(roc_auc_score(y_te,rf_proba,multi_class='ovr')),3)
        rf_cm    = confusion_matrix(y_te,rf_pred).tolist()

        fi = sorted(
            [{"feature":f,"importance":round(float(i),4)} for f,i in zip(fcols,rf.feature_importances_)],
            key=lambda x:x['importance'], reverse=True
        )

        # ROC curves
        roc_curves = {}
        for i,cls in enumerate(classes):
            fpr,tpr,_ = roc_curve((y_te==i).astype(int), rf_proba[:,i])
            roc_curves[cls] = {"fpr":fpr.tolist(),"tpr":tpr.tolist()}

        # ── Learning curves for both models ──
        def get_lc(estimator, Xd, yd):
            ts,tr,vl = learning_curve(estimator, Xd, yd,
                                      train_sizes=np.linspace(0.2,1.0,6),
                                      scoring='accuracy', cv=3)
            return {"train_sizes":ts.tolist(),
                    "train_acc":tr.mean(axis=1).round(3).tolist(),
                    "val_acc":  vl.mean(axis=1).round(3).tolist()}

        lc_lr = get_lc(LogisticRegression(max_iter=1000, random_state=42), X_tr_s, y_tr)
        lc_rf = get_lc(RandomForestClassifier(n_estimators=100,random_state=42), X_tr, y_tr)

        # Predictions on full set
        df['risk_pred']   = le.inverse_transform(rf.predict(X.fillna(0)))
        rp = rf.predict_proba(X.fillna(0))
        for i,cls in enumerate(classes):
            df[f'prob_{cls.lower()}'] = rp[:,i].round(3)

        customers_out = df[['ClientID','computed_risk','risk_pred',
                             'avg_payment_delay','payment_rate','nb_unpaid']
                          + [f'prob_{c.lower()}' for c in classes]].rename(
            columns={'ClientID':'id','computed_risk':'risk'}
        ).to_dict(orient='records')

        high_risk_n = int((df['risk_pred']=='High').sum())

        DASHBOARD_STATE['goal3'] = {
            "total_clients":     len(df),
            "high_risk_clients": high_risk_n,
            "high_risk_pct":     round(high_risk_n/len(df)*100,1),
            "lr_accuracy":lr_acc,"lr_auc":lr_auc,
            "rf_accuracy":rf_acc,"rf_auc":rf_auc,
            "best_model": "Random Forest" if rf_auc>lr_auc else "Logistic Regression",
            "top_risk_feature": fi[0]['feature'] if fi else "avg_payment_delay",
        }

        return jsonify({
            "customers":          customers_out,
            "class_distribution": df['risk_pred'].value_counts().to_dict(),
            "metrics": {
                "logistic_regression":{"accuracy":lr_acc,"roc_auc":lr_auc},
                "random_forest":      {"accuracy":rf_acc,"roc_auc":rf_auc},
            },
            "confusion_matrix":   {"lr":lr_cm,"rf":rf_cm,"classes":classes},
            "feature_importance": fi[:10],
            "roc_curves":         roc_curves,
            # ── NEW graph data ──
            "learning_curves":    {"lr":lc_lr,"rf":lc_rf},
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL 4 — Margin Prediction  (Ridge + XGBoost) — full graph data
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/goal4/margin', methods=['GET'])
def goal4_margin():
    try:
        c  = db()
        df = run_query(QUERY_GOAL4_MARGIN, c); c.close()

        df = df.dropna(subset=['gross_margin'])
        df['margin_pct']   = df['margin_pct'].fillna(0)
        df['price_markup'] = ((df['SellingPrice']-df['PurchasePrice'])
                               /df['PurchasePrice'].replace(0,np.nan)).fillna(0)

        fcols = [
            'PurchasePrice','SellingPrice','qty_sold','total_revenue',
            'avg_discount_pct','avg_tax','avg_on_hand','avg_safety_stock',
            'avg_stock_value','returns_qty','nb_customers','price_markup',
        ]
        df_m = df.dropna(subset=fcols)
        X    = df_m[fcols].fillna(0)
        y    = df_m['margin_pct']

        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=42)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)
        ridge_pred = ridge.predict(X_te_s)
        ridge_r2   = round(float(r2_score(y_te,ridge_pred)),4)
        ridge_mae  = round(float(mean_absolute_error(y_te,ridge_pred)),4)
        ridge_rmse = round(float(np.sqrt(mean_squared_error(y_te,ridge_pred))),4)

        # XGBoost
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        xgb.fit(X_tr, y_tr)
        xgb_pred = xgb.predict(X_te)
        xgb_r2   = round(float(r2_score(y_te,xgb_pred)),4)
        xgb_mae  = round(float(mean_absolute_error(y_te,xgb_pred)),4)
        xgb_rmse = round(float(np.sqrt(mean_squared_error(y_te,xgb_pred))),4)

        fi = sorted(
            [{"feature":f,"importance":round(float(i),4)} for f,i in zip(fcols,xgb.feature_importances_)],
            key=lambda x:x['importance'], reverse=True
        )

        # Full predictions
        df_m = df_m.copy()
        df_m['xgb_pred']   = xgb.predict(X.fillna(0)).round(2)
        df_m['ridge_pred'] = ridge.predict(sc.transform(X.fillna(0))).round(2)
        df_m['error_xgb']  = (df_m['margin_pct']-df_m['xgb_pred']).round(2)
        df_m['error_ridge']= (df_m['margin_pct']-df_m['ridge_pred']).round(2)
        df_m['risk']       = df_m['error_xgb'].abs().apply(lambda e:'High' if e>10 else 'Normal')

        cat_summary = {}
        for cat,sub in df_m.groupby('ProductCategory'):
            cat_summary[cat] = {
                "avg_actual":    round(float(sub['margin_pct'].mean()),2),
                "avg_predicted": round(float(sub['xgb_pred'].mean()),2),
                "count":         int(len(sub)),
            }

        products_out = df_m[['ProductID','ProductName','ProductCategory',
                              'margin_pct','xgb_pred','ridge_pred',
                              'error_xgb','error_ridge','risk']].rename(
            columns={'ProductID':'id','ProductName':'name',
                     'ProductCategory':'category','margin_pct':'actual_margin'}
        ).to_dict(orient='records')

        # ── Scatter: actual vs predicted (both models, test set) ──
        scatter = {
            "actual":       y_te.round(2).tolist(),
            "ridge_pred":   ridge_pred.round(2).tolist(),
            "xgb_pred":     xgb_pred.round(2).tolist(),
            "ridge_resid":  (y_te - ridge_pred).round(2).tolist(),
            "xgb_resid":    (y_te - xgb_pred).round(2).tolist(),
        }

        # ── Alpha search for Ridge (regularisation curve) ──
        alphas  = [0.01, 0.1, 1, 10, 100, 1000]
        ridge_alpha_curve = []
        for a in alphas:
            r = Ridge(alpha=a).fit(X_tr_s, y_tr)
            ridge_alpha_curve.append({
                "alpha": a,
                "r2":    round(float(r2_score(y_te, r.predict(X_te_s))),4),
                "rmse":  round(float(np.sqrt(mean_squared_error(y_te, r.predict(X_te_s)))),4),
            })

        # ── XGBoost: training loss curve ──
        # Use DataFrames (not .values) so feature names are preserved
        eval_set = [(X_tr, y_tr), (X_te, y_te)]
        xgb_eval = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                 random_state=42, eval_metric='rmse')
        xgb_eval.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        xgb_loss = {
            "iterations": list(range(1, 201)),
            "train_rmse": [round(v,4) for v in xgb_eval.evals_result()['validation_0']['rmse']],
            "val_rmse":   [round(v,4) for v in xgb_eval.evals_result()['validation_1']['rmse']],
        }

        high_risk_n = int((df_m['risk']=='High').sum())

        DASHBOARD_STATE['goal4'] = {
            "total_products":     len(df_m),
            "high_risk_products": high_risk_n,
            "avg_margin_pct":     round(float(df_m['margin_pct'].mean()),2),
            "ridge_r2":ridge_r2,"xgb_r2":xgb_r2,
            "best_model": "XGBoost" if xgb_r2>ridge_r2 else "Ridge",
            "top_margin_feature": fi[0]['feature'] if fi else "price_markup",
            "categories": list(cat_summary.keys()),
        }

        return jsonify({
            "products":         products_out,
            "category_summary": cat_summary,
            "metrics": {
                "ridge":   {"r2":ridge_r2,  "mae":ridge_mae,  "rmse":ridge_rmse},
                "xgboost": {"r2":xgb_r2,    "mae":xgb_mae,    "rmse":xgb_rmse},
            },
            "feature_importance": fi[:10],
            "high_risk_count":    high_risk_n,
            # ── NEW graph data ──
            "scatter":            scatter,
            "ridge_alpha_curve":  ridge_alpha_curve,
            "xgb_loss_curve":     xgb_loss,
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED A — Recommendations  (Item-Based CF + SVD)
# ═══════════════════════════════════════════════════════════════════════════════
QUERY_RECO = """
SELECT f.ClientID, f.ProductID, SUM(f.SoldItemsQty) AS qty
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus] f
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
  AND f.SoldItemsQty > 0
GROUP BY f.ClientID, f.ProductID
"""
QUERY_PRODUCT_NAMES = """
SELECT ProductID, ProductName, ProductCategory
FROM [dwh_parapharmacie].[dbo].[DIM_Product]
"""

@app.route('/api/advanced/recommendations', methods=['GET'])
def recommendations():
    try:
        client_id = request.args.get('client_id', None)
        c = db()
        df_s = run_query(QUERY_RECO,          c)
        df_p = run_query(QUERY_PRODUCT_NAMES, c); c.close()

        prod_info = df_p.set_index('ProductID').to_dict(orient='index')
        matrix    = df_s.pivot_table(index='ClientID',columns='ProductID',values='qty',fill_value=0)
        clients   = matrix.index.tolist()
        products  = matrix.columns.tolist()
        M         = matrix.values.astype(float)

        n_comp   = min(20, M.shape[1]-1)
        svd      = TruncatedSVD(n_components=n_comp, random_state=42)
        M_svd    = svd.fit_transform(M)
        M_rec    = M_svd @ svd.components_

        # ── SVD explained variance curve ──
        svd_variance = [
            {"k":k+1,"variance":round(float(svd.explained_variance_ratio_[:k+1].sum()),4),
             "individual":round(float(svd.explained_variance_ratio_[k]),4)}
            for k in range(n_comp)
        ]

        cidx = clients.index(client_id) if client_id and client_id in clients else 0
        client_id = clients[cidx]
        user_vec  = M[cidx]

        # Item-Based CF
        item_sims = cosine_similarity(M.T)
        scores_ib = np.zeros(len(products))
        for j,r in enumerate(user_vec):
            if r>0: scores_ib += r*item_sims[j]
        scores_ib[user_vec>0] = 0
        top_ib = np.argsort(scores_ib)[::-1][:5]
        item_based = [{"id":str(products[i]),
                       "name":prod_info.get(products[i],{}).get('ProductName',f'P{products[i]}'),
                       "category":prod_info.get(products[i],{}).get('ProductCategory','N/A'),
                       "score":round(float(scores_ib[i]),3)} for i in top_ib]

        # SVD-Based
        scores_svd = M_rec[cidx].copy()
        scores_svd[user_vec>0] = 0
        top_svd = np.argsort(scores_svd)[::-1][:5]
        svd_based = [{"id":str(products[i]),
                      "name":prod_info.get(products[i],{}).get('ProductName',f'P{products[i]}'),
                      "category":prod_info.get(products[i],{}).get('ProductCategory','N/A'),
                      "score":round(float(scores_svd[i]),3)} for i in top_svd]

        # ── Score distribution (all customers, top-1 SVD score) ──
        all_scores = []
        for i in range(len(clients)):
            sv = M_rec[i].copy(); sv[M[i]>0] = 0
            all_scores.append(round(float(sv.max()),3) if sv.max()>0 else 0)
        # Bin into histogram
        hist, edges = np.histogram(all_scores, bins=20)
        score_dist = [{"bin_center":round(float((edges[i]+edges[i+1])/2),3),
                       "count":int(hist[i])} for i in range(len(hist))]

        # ── Item-Item similarity heatmap (top-10 products by popularity) ──
        pop_idx = np.argsort(M.sum(axis=0))[::-1][:10]
        sim_matrix_top = item_sims[np.ix_(pop_idx, pop_idx)]
        heatmap = {
            "labels":  [str(products[i]) for i in pop_idx],
            "matrix":  sim_matrix_top.round(3).tolist(),
        }

        sparsity = round(float((M==0).sum()/M.size*100),1)
        DASHBOARD_STATE['advA'] = {
            "total_clients":len(clients),"total_products":len(products),
            "sparsity":sparsity,"svd_components":n_comp,
            "explained_var":round(float(svd.explained_variance_ratio_.sum()),3),
        }

        return jsonify({
            "client_id":            client_id,
            "item_based":           item_based,
            "svd_based":            svd_based,
            "svd_variance":         svd_variance,
            "matrix_shape":         {"customers":len(clients),"products":len(products),"sparsity":sparsity},
            "total_recommendations":len(clients)*5,
            # ── NEW graph data ──
            "score_distribution":   score_dist,
            "similarity_heatmap":   heatmap,
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED B — Anomaly Detection  (Isolation Forest + PCA-Autoencoder)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/advanced/anomalies', methods=['GET'])
def anomalies():
    try:
        c = db()
        df_ts   = run_query(QUERY_GOAL1_TIMESERIES, c)
        df_cred = run_query(QUERY_GOAL3_CREDIT,     c); c.close()

        df_ts['sale_date'] = pd.to_datetime(df_ts['sale_date'])
        dfw = (df_ts.groupby(pd.Grouper(key='sale_date',freq='W'))
                    .agg(total_revenue=('total_revenue','sum'),
                         total_qty_sold=('total_qty_sold','sum'))
                    .reset_index())
        dfw = dfw[dfw['total_revenue']>0].reset_index(drop=True)

        ts_feat    = dfw[['total_revenue','total_qty_sold']].fillna(0)
        iso_ts     = IsolationForest(contamination=0.08, random_state=42)
        iso_ts.fit(ts_feat)
        iso_scores = iso_ts.score_samples(ts_feat)
        iso_labels = (iso_ts.predict(ts_feat)==-1).tolist()

        # PCA-AE
        pca_ae  = PCA(n_components=1)
        recon   = pca_ae.inverse_transform(pca_ae.fit_transform(ts_feat))
        ae_errs = np.mean((ts_feat.values-recon)**2, axis=1)
        ae_thr  = np.percentile(ae_errs, 92)
        ae_labels = (ae_errs>ae_thr).tolist()
        consensus = [a and b for a,b in zip(iso_labels,ae_labels)]

        daily_series = []
        for i, (_, row) in enumerate(dfw.iterrows()):
            daily_series.append({
                "date":        row['sale_date'].strftime('%Y-%m-%d'),
                "revenue":     round(float(row['total_revenue']),2),
                "qty":         int(row['total_qty_sold']),
                "iso_anomaly": iso_labels[i],
                "ae_anomaly":  ae_labels[i],
                "consensus":   consensus[i],
                "iso_score":   round(float(iso_scores[i]),4),
                "ae_error":    round(float(ae_errs[i]),5),
            })

        # Payment anomalies
        pay_feat   = df_cred[['avg_payment_delay','max_payment_delay',
                               'payment_rate','nb_unpaid']].fillna(0)
        iso_pay    = IsolationForest(contamination=0.08, random_state=42)
        iso_pay.fit(pay_feat)
        pay_scores = iso_pay.score_samples(pay_feat)
        pay_labels = (iso_pay.predict(pay_feat)==-1).tolist()

        customer_payments = []
        for i,(_,row) in enumerate(df_cred.iterrows()):
            customer_payments.append({
                "id":              str(row['ClientID']),
                "avg_delay":       round(float(row['avg_payment_delay']),1),
                "nb_unpaid":       int(row['nb_unpaid']),
                "payment_rate":    round(float(row['payment_rate']),3),
                "payment_anomaly": pay_labels[i],
                "score":           round(float(pay_scores[i]),3),
            })

        # ── IF score histogram ──
        hist_ts, edges_ts = np.histogram(iso_scores, bins=30)
        iso_hist = [{"score":round(float((edges_ts[j]+edges_ts[j+1])/2),4),
                     "count":int(hist_ts[j])} for j in range(len(hist_ts))]

        hist_pay, edges_pay = np.histogram(pay_scores, bins=30)
        pay_hist = [{"score":round(float((edges_pay[j]+edges_pay[j+1])/2),4),
                     "count":int(hist_pay[j])} for j in range(len(hist_pay))]

        # ── AE reconstruction error over time ──
        ae_curve = [{"date":dfw.iloc[i]['sale_date'].strftime('%Y-%m-%d'),
                     "ae_error":round(float(ae_errs[i]),5),
                     "threshold":round(float(ae_thr),5),
                     "is_anomaly":ae_labels[i]} for i in range(len(dfw))]

        DASHBOARD_STATE['advB'] = {
            "total_weeks":       len(dfw),
            "iso_anomalies":     int(sum(iso_labels)),
            "ae_anomalies":      int(sum(ae_labels)),
            "consensus":         int(sum(consensus)),
            "payment_anomalies": int(sum(pay_labels)),
        }

        return jsonify({
            "daily_series":      daily_series,
            "customer_payments": customer_payments,
            "summary":           DASHBOARD_STATE['advB'],
            # ── NEW graph data ──
            "iso_score_histogram":  iso_hist,
            "pay_score_histogram":  pay_hist,
            "ae_reconstruction_curve": ae_curve,
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard KPIs
# ═══════════════════════════════════════════════════════════════════════════════
QUERY_KPIS = """
SELECT
    SUM(f.TotalSales)                               AS total_revenue,
    COUNT(DISTINCT f.ClientID)                      AS total_customers,
    AVG(CASE WHEN f.CostVariance>0
        THEN (f.TotalSales-f.CostVariance)/f.CostVariance*100
        ELSE NULL END)                              AS avg_margin_pct,
    COUNT(DISTINCT f.DocumentNumber)                AS nb_transactions,
    SUM(CASE WHEN f.IsPayed=0 THEN 1 ELSE 0 END)   AS nb_unpaid
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus] f
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
"""

@app.route('/api/dashboard/kpis', methods=['GET'])
def dashboard_kpis():
    try:
        c  = db()
        df = run_query(QUERY_KPIS, c); c.close()
        row = df.iloc[0]

        kpis = {
            "total_revenue":    {"value":round(float(row['total_revenue']  or 0),2),"unit":"TND"},
            "total_customers":  {"value":int(row['total_customers']         or 0),  "unit":""},
            "avg_margin":       {"value":round(float(row['avg_margin_pct'] or 0),1),"unit":"%"},
            "nb_transactions":  {"value":int(row['nb_transactions']         or 0),  "unit":""},
            "nb_unpaid":        {"value":int(row['nb_unpaid']               or 0),  "unit":""},
            "anomalies_today":  {"value":DASHBOARD_STATE['advB'].get('consensus',0),"unit":""},
            "high_risk_clients":{"value":DASHBOARD_STATE['goal3'].get('high_risk_clients',0),"unit":""},
        }
        xm = DASHBOARD_STATE['goal1'].get('xgb_mape')
        if xm: kpis["forecast_accuracy"] = {"value":round(100-xm,1),"unit":"%"}

        DASHBOARD_STATE['kpis'] = {k:v['value'] for k,v in kpis.items()}
        return jsonify(kpis)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG — Data Snapshot
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/api/debug/data-snapshot', methods=['GET'])
def data_snapshot():
    try:
        c = db()
        df1 = run_query(QUERY_GOAL1_TIMESERIES,   c)
        df2 = run_query(QUERY_GOAL2_SEGMENTATION, c)
        df3 = run_query(QUERY_GOAL3_CREDIT,       c)
        df4 = run_query(QUERY_GOAL4_MARGIN,       c)
        c.close()
        return jsonify({
            "goal1":{"row_count":len(df1),
                     "date_range":[str(df1['sale_date'].min()),str(df1['sale_date'].max())],
                     "total_revenue":float(df1['total_revenue'].sum()),
                     "sample":df1.head(3).to_dict(orient='records')},
            "goal2":{"row_count":len(df2),
                     "unique_clients":int(df2['ClientID'].nunique()),
                     "avg_revenue":round(float(df2['total_revenue'].mean()),2),
                     "sample":df2.head(3).to_dict(orient='records')},
            "goal3":{"row_count":len(df3),
                     "risk_dist":df3['computed_risk'].value_counts().to_dict(),
                     "sample":df3.head(3).to_dict(orient='records')},
            "goal4":{"row_count":len(df4),
                     "unique_products":int(df4['ProductID'].nunique()),
                     "avg_margin":round(float(df4['margin_pct'].dropna().mean()),2),
                     "categories":df4['ProductCategory'].value_counts().to_dict(),
                     "sample":df4.head(3).to_dict(orient='records')},
        })
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# CHATBOT  (Claude-powered, grounded in live DASHBOARD_STATE)
# ═══════════════════════════════════════════════════════════════════════════════
def build_context_summary():
    s = DASHBOARD_STATE; parts = []
    if s['kpis']:
        k=s['kpis']
        parts.append(f"KPIs: revenue={k.get('total_revenue')} TND, customers={k.get('total_customers')}, margin={k.get('avg_margin')}%, unpaid={k.get('nb_unpaid')}.")
    if s['goal1']:
        g=s['goal1']
        parts.append(f"Goal1-Forecasting: {g.get('weeks_of_data')} weeks, avg_weekly_rev={g.get('avg_weekly_revenue')} TND, best_model={g.get('best_model')} MAPE={g.get('xgb_mape')}%, 4w_forecast={g.get('forecast_4w')} TND.")
    if s['goal2']:
        g=s['goal2']; sc=g.get('segment_counts',{})
        parts.append(f"Goal2-Segmentation: {g.get('total_customers')} customers, best_k={g.get('best_k')} (silhouette={g.get('best_sil')}), segments={sc}.")
    if s['goal3']:
        g=s['goal3']
        parts.append(f"Goal3-CreditRisk: {g.get('total_clients')} clients, high_risk={g.get('high_risk_clients')} ({g.get('high_risk_pct')}%), best={g.get('best_model')} AUC={g.get('rf_auc')}, top_driver={g.get('top_risk_feature')}.")
    if s['goal4']:
        g=s['goal4']
        parts.append(f"Goal4-Margin: {g.get('total_products')} products, avg_margin={g.get('avg_margin_pct')}%, high_risk={g.get('high_risk_products')}, best={g.get('best_model')} R²={g.get('xgb_r2')}, top_feat={g.get('top_margin_feature')}.")
    if s['advA']:
        g=s['advA']
        parts.append(f"AdvA-Recommendations: {g.get('total_clients')}×{g.get('total_products')} matrix, sparsity={g.get('sparsity')}%, SVD_var={g.get('explained_var')}.")
    if s['advB']:
        g=s['advB']
        parts.append(f"AdvB-Anomalies: {g.get('total_weeks')} weeks, iso={g.get('iso_anomalies')}, ae={g.get('ae_anomalies')}, consensus={g.get('consensus')}, payment_anomalies={g.get('payment_anomalies')}.")
    return "\n".join(parts) or "No data loaded yet."

def classify_intent(msg):
    m=msg.lower()
    if any(w in m for w in ['forecast','sarima','xgb','revenue','sales','weekly','trend']): return 'goal1'
    if any(w in m for w in ['segment','cluster','kmeans','dbscan','champion','loyal','dormant','rfm','elbow','silhouette']): return 'goal2'
    if any(w in m for w in ['credit','risk','payment','delay','unpaid','logistic','random forest']): return 'goal3'
    if any(w in m for w in ['margin','price','ridge','markup','cost','gross']): return 'goal4'
    if any(w in m for w in ['recommend','svd','collabor','item','suggest']): return 'advA'
    if any(w in m for w in ['anomal','detect','isolation','autoencoder','unusual','outlier']): return 'advB'
    return 'general'

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data    = request.json or {}
    message = data.get('message','').strip()
    if not message: return jsonify({"error":"empty message"}), 400

    context = build_context_summary()
    intent  = classify_intent(message)
    focus   = {
        'goal1':"Focus on Goal1 – Sales Forecasting (SARIMA & XGBoost).",
        'goal2':"Focus on Goal2 – Customer Segmentation (silhouette-optimized K-Means & DBSCAN).",
        'goal3':"Focus on Goal3 – Credit Risk (Logistic Regression & Random Forest).",
        'goal4':"Focus on Goal4 – Margin Prediction (Ridge & XGBoost).",
        'advA': "Focus on AdvancedA – Product Recommendations (Item-Based CF & SVD).",
        'advB': "Focus on AdvancedB – Anomaly Detection (Isolation Forest & PCA-Autoencoder).",
        'general':"Give an overview or answer the general question.",
    }.get(intent,'')

    system_prompt = f"""You are PharmAI Assistant — a senior data-science analyst for dwh_parapharmacie.
Answer questions grounded in the LIVE DASHBOARD DATA below. Cite exact numbers. Be concise and professional.
If a section shows no data, say "Load the [page] first."

LIVE DATA:
{context}

{focus}"""

    try:
        import requests as rl
        resp = rl.post("https://api.anthropic.com/v1/messages",
                       headers={"Content-Type":"application/json"},
                       json={"model":"claude-sonnet-4-20250514","max_tokens":512,
                             "system":system_prompt,
                             "messages":[{"role":"user","content":message}]},
                       timeout=30)
        reply = resp.json().get("content",[{}])[0].get("text","")
    except Exception:
        s = DASHBOARD_STATE
        fallbacks = {
            'goal2': f"Segmentation: best k={s['goal2'].get('best_k','?')} (silhouette={s['goal2'].get('best_sil','?')}). Segments: {s['goal2'].get('segment_counts','?')}." if s['goal2'] else "Load the Segmentation page first.",
            'goal1': f"Forecasting: avg weekly revenue={s['goal1'].get('avg_weekly_revenue','?')} TND, best model={s['goal1'].get('best_model','?')} MAPE={s['goal1'].get('xgb_mape','?')}%." if s['goal1'] else "Load the Forecasting page first.",
            'goal3': f"Credit: {s['goal3'].get('high_risk_clients','?')} high-risk ({s['goal3'].get('high_risk_pct','?')}%), AUC={s['goal3'].get('rf_auc','?')}." if s['goal3'] else "Load the Credit Risk page first.",
            'goal4': f"Margin: avg={s['goal4'].get('avg_margin_pct','?')}%, XGBoost R²={s['goal4'].get('xgb_r2','?')}." if s['goal4'] else "Load the Margin page first.",
            'advA':  f"Recommendations: {s['advA'].get('total_clients','?')}×{s['advA'].get('total_products','?')} matrix." if s['advA'] else "Load Recommendations first.",
            'advB':  f"Anomalies: {s['advB'].get('consensus','?')} consensus anomalies detected." if s['advB'] else "Load Anomaly Detection first.",
            'general': context,
        }
        reply = fallbacks.get(intent, context)

    return jsonify({"response":reply,"timestamp":datetime.now().isoformat(),
                    "model":"PharmAI Assistant (Claude-powered)","intent":intent})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
