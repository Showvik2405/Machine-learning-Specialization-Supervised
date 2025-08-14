// crazy_lr_logreg.cpp
// Single-file, zero-dependency (STL-only) implementations of:
//   - LinearRegression (MSE, R^2, L2 regularization, feature scaling)
//   - LogisticRegression (binary, cross-entropy, L2 regularization)
//   - Tiny utilities: dataset generation, train/test split, standardization
// Build:  g++ -std=gnu++17 -O3 crazy_lr_logreg.cpp -o crazy && ./crazy
// Note: This is educational (readable) rather than maximally optimized.

#include <bits/stdc++.h>
using namespace std;

// ---------------------- helpers ----------------------
static inline double dot(const vector<double>& a, const vector<double>& b){
    double s=0.0; size_t n=a.size();
    for(size_t i=0;i<n;++i) s += a[i]*b[i];
    return s;
}

static inline void axpy(double a, const vector<double>& x, vector<double>& y){
    // y += a * x
    size_t n=x.size();
    for(size_t i=0;i<n;++i) y[i] += a * x[i];
}

static inline double sigmoid(double z){
    if (z>=0){
        double ez = exp(-z);
        return 1.0/(1.0+ez);
    } else {
        double ez = exp(z);
        return ez/(1.0+ez);
    }
}

struct StandardScaler{
    vector<double> mean, stdev;
    bool fitted=false;
    void fit(const vector<vector<double>>& X){
        size_t n=X.size(); if(n==0){fitted=false; return;} size_t d=X[0].size();
        mean.assign(d,0.0); stdev.assign(d,0.0);
        for(const auto& x: X) for(size_t j=0;j<d;++j) mean[j]+=x[j];
        for(size_t j=0;j<d;++j) mean[j]/=double(n);
        for(const auto& x: X) for(size_t j=0;j<d;++j){ double v=x[j]-mean[j]; stdev[j]+=v*v; }
        for(size_t j=0;j<d;++j){ stdev[j]=sqrt(stdev[j]/max<size_t>(1,n-1)); if(stdev[j]==0) stdev[j]=1.0; }
        fitted=true;
    }
    vector<vector<double>> transform(const vector<vector<double>>& X) const{
        if(!fitted) return X; size_t n=X.size(); if(n==0) return X; size_t d=X[0].size();
        vector<vector<double>> Z(n, vector<double>(d));
        for(size_t i=0;i<n;++i) for(size_t j=0;j<d;++j) Z[i][j]=(X[i][j]-mean[j])/stdev[j];
        return Z;
    }
};

struct TrainTestSplit{
    vector<vector<double>> X_train, X_test; 
    vector<double> y_train, y_test;
};

TrainTestSplit train_test_split(const vector<vector<double>>& X, const vector<double>& y, double test_size=0.25, unsigned seed=42){
    size_t n=X.size(); vector<size_t> idx(n); iota(idx.begin(), idx.end(), 0);
    mt19937 rng(seed); shuffle(idx.begin(), idx.end(), rng);
    size_t n_test = size_t(round(test_size*n));
    TrainTestSplit out; out.X_test.reserve(n_test); out.y_test.reserve(n_test);
    out.X_train.reserve(n-n_test); out.y_train.reserve(n-n_test);
    for(size_t k=0;k<n;++k){
        size_t i=idx[k];
        if(k<n_test){ out.X_test.push_back(X[i]); out.y_test.push_back(y[i]); }
        else { out.X_train.push_back(X[i]); out.y_train.push_back(y[i]); }
    }
    return out;
}

static inline vector<double> add_bias(const vector<double>& x){
    vector<double> xb(x.size()+1,1.0); // bias at index 0 = 1
    for(size_t j=0;j<x.size();++j) xb[j+1]=x[j];
    return xb;
}

// ---------------------- Linear Regression ----------------------
class LinearRegression{
public:
    vector<double> w; // includes bias w0 at index 0
    double alpha=0.05; // learning rate
    int epochs=400;    // iterations
    double lambda=0.0; // L2
    bool fit_intercept=true;

    LinearRegression(double lr=0.05,int ep=400,double l2=0.0,bool intercept=true)
        : w(), alpha(lr), epochs(ep), lambda(l2), fit_intercept(intercept) {}

    void fit(const vector<vector<double>>& X_in, const vector<double>& y){
        size_t n=X_in.size(); if(n==0) return; size_t d=X_in[0].size();
        // Build design with/without bias
        vector<vector<double>> X(n);
        for(size_t i=0;i<n;++i){ X[i] = fit_intercept ? add_bias(X_in[i]) : X_in[i]; }
        size_t p = X[0].size();
        w.assign(p, 0.0);
        // Gradient descent
        for(int it=0; it<epochs; ++it){
            vector<double> grad(p, 0.0);
            for(size_t i=0;i<n;++i){
                double yhat = dot(w, X[i]);
                double err = yhat - y[i];
                // grad += err * x_i
                axpy(err, X[i], grad);
            }
            // average and add L2 (skip bias for reg)
            for(size_t j=0;j<p;++j){
                grad[j] /= double(n);
                if(lambda>0.0){ if(!(fit_intercept && j==0)) grad[j] += lambda * w[j]; }
                w[j] -= alpha * grad[j];
            }
            // simple cosine annealing-ish tweak to keep it fun
            if((it+1)%100==0) alpha *= 0.7;
        }
    }

    vector<double> predict(const vector<vector<double>>& X_in) const{
        size_t n=X_in.size(); vector<double> out(n);
        bool intercept=fit_intercept;
        for(size_t i=0;i<n;++i){
            if(intercept){ auto xb=add_bias(X_in[i]); out[i] = dot(w, xb); }
            else out[i] = dot(w, X_in[i]);
        }
        return out;
    }

    static double mse(const vector<double>& y, const vector<double>& yhat){
        double s=0; size_t n=y.size(); for(size_t i=0;i<n;++i){ double e=yhat[i]-y[i]; s+=e*e; }
        return s/double(max<size_t>(1,n));
    }

    static double r2(const vector<double>& y, const vector<double>& yhat){
        double mean=accumulate(y.begin(), y.end(), 0.0)/double(y.size());
        double ss_res=0, ss_tot=0; 
        for(size_t i=0;i<y.size();++i){ double e=yhat[i]-y[i]; ss_res+=e*e; double v=y[i]-mean; ss_tot+=v*v; }
        return 1.0 - (ss_res/max(1e-12, ss_tot));
    }
};

// ---------------------- Logistic Regression (Binary) ----------------------
class LogisticRegression{
public:
    vector<double> w; // includes bias at index 0 if fit_intercept
    double alpha=0.1; int epochs=500; double lambda=0.0; bool fit_intercept=true;
    LogisticRegression(double lr=0.1,int ep=500,double l2=0.0,bool intercept=true)
        : w(), alpha(lr), epochs(ep), lambda(l2), fit_intercept(intercept) {}

    void fit(const vector<vector<double>>& X_in, const vector<double>& y){
        size_t n=X_in.size(); if(n==0) return; size_t d=X_in[0].size();
        vector<vector<double>> X(n);
        for(size_t i=0;i<n;++i){ X[i] = fit_intercept ? add_bias(X_in[i]) : X_in[i]; }
        size_t p=X[0].size(); w.assign(p, 0.0);
        for(int it=0; it<epochs; ++it){
            vector<double> grad(p, 0.0);
            for(size_t i=0;i<n;++i){
                double z = dot(w, X[i]);
                double pr = sigmoid(z);
                double err = pr - y[i];
                axpy(err, X[i], grad);
            }
            for(size_t j=0;j<p;++j){
                grad[j] /= double(n);
                if(lambda>0.0){ if(!(fit_intercept && j==0)) grad[j] += lambda * w[j]; }
                w[j] -= alpha * grad[j];
            }
            if((it+1)%150==0) alpha *= 0.7;
        }
    }

    vector<double> predict_proba(const vector<vector<double>>& X_in) const{
        size_t n=X_in.size(); vector<double> out(n);
        bool intercept=fit_intercept;
        for(size_t i=0;i<n;++i){ double z = intercept ? dot(w, add_bias(X_in[i])) : dot(w, X_in[i]); out[i]=sigmoid(z); }
        return out;
    }

    vector<int> predict(const vector<vector<double>>& X_in, double threshold=0.5) const{
        vector<double> p = predict_proba(X_in); vector<int> yhat(p.size());
        for(size_t i=0;i<p.size();++i) yhat[i] = (p[i]>=threshold)?1:0; return yhat;
    }

    static double accuracy(const vector<int>& y, const vector<int>& yhat){
        size_t n=y.size(); size_t c=0; for(size_t i=0;i<n;++i) if(y[i]==yhat[i]) ++c; return double(c)/double(max<size_t>(1,n));
    }
};

// ---------------------- Feature engineering ----------------------
vector<vector<double>> polynomial_features(const vector<vector<double>>& X, int degree){
    // Generate [x1, x2, ..., xk] -> all monomials up to 'degree' (without bias)
    // For k up to ~5 and degree up to 3 it's fine.
    size_t n=X.size(); if(n==0) return X; size_t k=X[0].size();
    // Generate exponent tuples via recursion
    vector<vector<int>> exps; 
    function<void(size_t,int,vector<int>&)> gen = [&](size_t idx,int left,vector<int>& cur){
        if(idx==k-1){ cur[idx]=left; exps.push_back(cur); return; }
        for(int e=0;e<=left;++e){ cur[idx]=e; gen(idx+1, left-e, cur); }
    };
    vector<vector<double>> Z; Z.reserve(n);
    for(int d=1; d<=degree; ++d){
        vector<int> cur(k,0); exps.clear(); gen(0,d,cur);
        if(d==1 && k==X[0].size()){
            // we'll append progressively below
        }
        if(Z.empty()){
            Z.assign(n, {});
        }
        for(size_t i=0;i<n;++i){
            for(const auto& e: exps){
                double m=1.0; for(size_t j=0;j<k;++j){ if(e[j]==0) continue; m*= pow(X[i][j], e[j]); }
                Z[i].push_back(m);
            }
        }
    }
    return Z;
}

// ---------------------- Synthetic datasets ----------------------
struct Dataset{ vector<vector<double>> X; vector<double> y; };

Dataset make_linear(size_t n=400, unsigned seed=123){
    mt19937 rng(seed); normal_distribution<double> noise(0.0, 0.6);
    uniform_real_distribution<double> u(-3.0, 3.0);
    vector<vector<double>> X(n, vector<double>(2)); vector<double> y(n);
    for(size_t i=0;i<n;++i){ X[i][0]=u(rng); X[i][1]=u(rng); y[i]=3.0*X[i][0] - 2.0*X[i][1] + 1.5 + noise(rng); }
    return {X,y};
}

Dataset make_logistic(size_t n=600, unsigned seed=321){
    mt19937 rng(seed);
    normal_distribution<double> c0x(-1.5, 1.0), c0y(0.0, 1.0);
    normal_distribution<double> c1x(1.5, 1.0),  c1y(0.5, 1.0);
    vector<vector<double>> X; vector<double> y; X.reserve(n); y.reserve(n);
    for(size_t i=0;i<n/2;++i){ X.push_back({c0x(rng), c0y(rng)}); y.push_back(0.0); }
    for(size_t i=0;i<n/2;++i){ X.push_back({c1x(rng), c1y(rng)}); y.push_back(1.0); }
    return {X,y};
}

// ---------------------- main demo ----------------------
int main(){
    ios::sync_with_stdio(false);

    cout << "\n===== LINEAR REGRESSION (with scaling + poly features + L2) =====\n";
    auto dlin = make_linear();
    StandardScaler scaler; scaler.fit(dlin.X); auto Xs = scaler.transform(dlin.X);
    auto Xpoly = polynomial_features(Xs, 2); // quadratic terms for some madness
    auto split1 = train_test_split(Xpoly, dlin.y, 0.25, 7);
    LinearRegression lr(0.1, 800, 0.01, true);
    lr.fit(split1.X_train, split1.y_train);
    auto yhat = lr.predict(split1.X_test);
    cout << fixed << setprecision(4);
    cout << "MSE: " << LinearRegression::mse(split1.y_test, yhat) << "\n";
    cout << "R^2: " << LinearRegression::r2(split1.y_test, yhat) << "\n";
    cout << "Weights (w0..): "; for(double wi: lr.w) cout << wi << ' '; cout << "\n";

    cout << "\n===== LOGISTIC REGRESSION (binary, L2) =====\n";
    auto dlog = make_logistic();
    StandardScaler scaler2; scaler2.fit(dlog.X); auto X2 = scaler2.transform(dlog.X);
    auto split2 = train_test_split(X2, dlog.y, 0.3, 9);
    LogisticRegression clf(0.2, 700, 0.05, true);
    clf.fit(split2.X_train, split2.y_train);
    auto yprob = clf.predict_proba(split2.X_test);
    vector<int> ytrue(split2.y_test.size()); for(size_t i=0;i<ytrue.size();++i) ytrue[i]=int(split2.y_test[i]);
    auto ypred = clf.predict(split2.X_test, 0.5);

    cout << "Accuracy: " << LogisticRegression::accuracy(ytrue, ypred) << "\n";
    // quick precision/recall just for flair
    size_t tp=0, fp=0, fn=0, tn=0; 
    for(size_t i=0;i<ytrue.size();++i){ if(ytrue[i]==1 && ypred[i]==1) ++tp; else if(ytrue[i]==0 && ypred[i]==1) ++fp; else if(ytrue[i]==1 && ypred[i]==0) ++fn; else ++tn; }
    double precision = tp / max(1.0, double(tp+fp));
    double recall    = tp / max(1.0, double(tp+fn));
    cout << "Precision: " << precision << ", Recall: " << recall << "\n";
    cout << "Weights (w0..): "; for(double wi: clf.w) cout << wi << ' '; cout << "\n\n";

    
    return 0;
}
// Note: This code is educational and may not be optimized for production use.
// Feel free to modify, extend, or use it as a learning resource.