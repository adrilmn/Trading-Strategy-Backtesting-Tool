import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class BaseStrategy:
    """Classe abstraite de base pour les stratégies."""
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Doit être implémenté par la classe fille.")

class MLRidgePCAStrategy(BaseStrategy):
    """
    Stratégie Machine Learning : PCA + Régression Ridge.
    Utilise des fenêtres glissantes pour l'entraînement (Walk-Forward).
    """
    def __init__(self, train_window: int = 500, recalibration_step: int = 20, pca_components: int = 3):
        self.train_window = train_window            # Taille de l'historique d'entraînement
        self.recalibration_step = recalibration_step # Ré-entraînement tous les X jours
        self.pca = PCA(n_components=pca_components)
        self.model = Ridge(alpha=1.0)               # Régularisation L2
        self.scaler = StandardScaler()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1. Feature Engineering (Ingénierie des caractéristiques)
        df = pd.DataFrame(index=data.index)
        df['Ret_1'] = data['Close'].pct_change(1)
        df['Ret_3'] = data['Close'].pct_change(3)
        df['Ret_10'] = data['Close'].pct_change(10)
        df['Vol_20'] = df['Ret_1'].rolling(20).std()
        df['SMA_Dist'] = (data['Close'] - data['Close'].rolling(50).mean()) / data['Close'].rolling(50).mean()
        
        # Target : Le rendement du jour SUIVANT
        df['Target'] = df['Ret_1'].shift(-1)
        
        # Nettoyage des NaN initiaux liés aux fenêtres roulantes
        df.dropna(inplace=True)
        
        features = ['Ret_1', 'Ret_3', 'Ret_10', 'Vol_20', 'SMA_Dist']
        X = df[features].values
        y = df['Target'].values
        
        signals = pd.Series(index=data.index, data=0.0)
        
        # 2. Walk-Forward Training Loop
        # On avance par pas de `recalibration_step` pour optimiser le temps de calcul
        for i in range(self.train_window, len(df), self.recalibration_step):
            # Définition des bornes d'entraînement
            X_train = X[i - self.train_window : i]
            y_train = y[i - self.train_window : i]
            
            # Définition de la fenêtre de prédiction (Out-of-Sample)
            test_end = min(i + self.recalibration_step, len(df))
            X_test = X[i : test_end]
            
            # Pipeline ML
            # a. Standardisation (Crucial pour la PCA et la régularisation Ridge)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # b. Réduction de dimensionnalité (PCA)
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
            
            # c. Entraînement et Prédiction
            self.model.fit(X_train_pca, y_train)
            predictions = self.model.predict(X_test_pca)
            
            # d. Règle de décision probabiliste simple : 
            # Si le modèle prédit un rendement positif -> Long (1), sinon -> Short (-1)
            pred_signals = np.where(predictions > 0, 1.0, -1.0)
            
            # Injection des signaux dans la série globale
            current_dates = df.index[i : test_end]
            signals.loc[current_dates] = pred_signals
            
        return signals

class AdvancedVectorizedBacktester:
    """
    Moteur de backtest gérant les frictions non-linéaires :
    - Coût d'emprunt pour la vente à découvert (Shorting costs).
    - Slippage dynamique modélisé via la volatilité locale.
    """
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, 
                 initial_capital: float = 100000.0, 
                 base_tc: float = 0.0005,         # Frais fixes de courtage (0.05%)
                 slippage_factor: float = 0.05,   # Sensibilité du slippage à la volatilité
                 short_borrow_rate: float = 0.04): # Taux d'emprunt annuel pour shorter (4%)
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.base_tc = base_tc
        self.slippage_factor = slippage_factor
        self.short_borrow_rate = short_borrow_rate
        self.results = None

    def run(self) -> dict:
        self.data['Signal'] = self.strategy.generate_signals(self.data)
        
        # Décalage des signaux pour éviter le Look-Ahead Bias
        self.data['Position'] = self.data['Signal'].shift(1).fillna(0)
        self.data['Market_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Proxy de profondeur de marché : on utilise l'écart-type glissant.
        # Plus la volatilité est élevée, plus le carnet d'ordres est "creux", plus le slippage augmente.
        rolling_volatility = self.data['Market_Returns'].rolling(window=20).std().fillna(0)
        dynamic_tc = self.base_tc + (rolling_volatility * self.slippage_factor)
        
        self.data['Strategy_Returns'] = self.data['Position'] * self.data['Market_Returns']
        
        # Déduction des coûts de transaction dynamiques aux points de bascule
        trades = self.data['Position'].diff().fillna(0).abs()
        self.data['Strategy_Returns'] -= trades * dynamic_tc
        
        # Pénalité asymétrique : Coût de la vente à découvert
        # On déduit un prorata journalier du taux d'emprunt (borrow fee) sur les positions négatives
        daily_short_rate = self.short_borrow_rate / 252
        is_short = self.data['Position'] < 0
        self.data.loc[is_short, 'Strategy_Returns'] -= daily_short_rate
        
        # Calcul des courbes d'équité
        self.data['Market_Equity'] = self.initial_capital * np.exp(self.data['Market_Returns'].cumsum())
        self.data['Strategy_Equity'] = self.initial_capital * np.exp(self.data['Strategy_Returns'].cumsum())
        
        self.results = self.data.dropna()
        return self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        strat_ret = self.results['Strategy_Returns']
        ann_return = strat_ret.mean() * 252
        ann_vol = strat_ret.std() * np.sqrt(252)
        sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
        
        cum_returns = np.exp(strat_ret.cumsum())
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "Rendement Total": f"{(cum_returns.iloc[-1] - 1):.2%}",
            "Rendement Annualisé": f"{ann_return:.2%}",
            "Volatilité Annualisée": f"{ann_vol:.2%}",
            "Ratio de Sharpe": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }

    def plot_performance(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(self.results.index, self.results['Market_Equity'], label='Buy & Hold (Market)', color='gray', alpha=0.6)
        ax1.plot(self.results.index, self.results['Strategy_Equity'], label='ML Ridge PCA Strategy', color='navy')
        ax1.set_title("Performance avec Frictions (Slippage Dynamique + Short Fees)")
        ax1.set_ylabel('Capital (\$)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.results.index, self.results['Position'], color='darkorange', step='post')
        ax2.set_title('Positionnement (Walk-Forward ML)')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short (-1)', 'Neutre (0)', 'Long (1)'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Exécution

if __name__ == "__main__":
    # Génération de données synthétiques (Mouvement Brownien Géométrique)
    def generate_synthetic_data(days=2000, S0=100, mu=0.05, sigma=0.2):
        dt = 1/252
        prices = [S0]
        for _ in range(days):
            shock = np.random.normal(0, 1)
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock))
        dates = pd.date_range(start='2018-01-01', periods=days+1, freq='B')
        return pd.DataFrame({'Close': prices}, index=dates)

    print("Génération des données...")
    df = generate_synthetic_data(days=2500)

    # Note théorique importante : Puisque les données synthétiques suivent un Mouvement 
    # Brownien Géométrique pur (processus de Markov sans mémoire), la régression Ridge va 
    # mathématiquement fitter du bruit. En conditions réelles (sur un vrai dataset via yfinance), 
    # c'est là que l'algorithme identifiera les autocorrélations ou l'alpha caché.
    
    print("Initialisation du modèle ML (PCA + Ridge) avec entraînement par fenêtres glissantes...")
    ml_strategy = MLRidgePCAStrategy(train_window=500, recalibration_step=20, pca_components=3)

    print("Exécution du backtest avec frictions avancées...")
    backtester = AdvancedVectorizedBacktester(
        data=df, 
        strategy=ml_strategy, 
        initial_capital=100000.0,
        base_tc=0.0002,           # 2 bps de frais fixes
        slippage_factor=0.08,     # Forte pénalité en cas de pic de volatilité
        short_borrow_rate=0.05    # 5% d'intérêt annuel pour shorter
    )
    
    metrics = backtester.run()
    
    print("\n--- Métriques de Performance ---")
    for key, value in metrics.items():
        print(f"{key:25} : {value}")

    backtester.plot_performance()