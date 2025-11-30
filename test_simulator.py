"""
Tests for the retirement portfolio simulator.
"""
import numpy as np
import pandas as pd
import pytest

from simulator import run_simulation, calculate_statistics, _source_funds, MeanRevertingMarket, RandomWalkMarket, BlockBootstrapMarket, create_ar_model


class TestSourceFunds:
    """Test the fund sourcing logic."""
    
    def test_panic_market_uses_cash_first(self):
        """In panic markets, should use cash before equity."""
        result = _source_funds(
            desired=100_000,
            market_return=-0.20,  # Below panic threshold
            panic_threshold=-0.15,
            equity=500_000,
            cash=200_000
        )
        assert result == 100_000
    
    def test_panic_market_drains_cash_then_equity(self):
        """If cash insufficient in panic, use equity for remainder."""
        result = _source_funds(
            desired=150_000,
            market_return=-0.20,
            panic_threshold=-0.15,
            equity=500_000,
            cash=100_000  # Only 100k in cash
        )
        assert result == 150_000  # 100k from cash + 50k from equity
    
    def test_normal_market_uses_equity_first(self):
        """In normal markets, should use equity before cash."""
        result = _source_funds(
            desired=100_000,
            market_return=0.05,  # Positive return
            panic_threshold=-0.15,
            equity=500_000,
            cash=200_000
        )
        assert result == 100_000
    
    def test_no_equity_uses_cash(self):
        """When equity depleted, should use cash."""
        result = _source_funds(
            desired=100_000,
            market_return=0.05,
            panic_threshold=-0.15,
            equity=0,
            cash=200_000
        )
        assert result == 100_000
    
    def test_limited_funds_caps_withdrawal(self):
        """Cannot withdraw more than available."""
        result = _source_funds(
            desired=100_000,
            market_return=0.05,
            panic_threshold=-0.15,
            equity=30_000,
            cash=20_000
        )
        assert result == 50_000  # 30k equity + 20k cash

    def test_negative_equity_does_not_inflate_withdrawal(self):
        """Negative equity should not count toward available funds."""
        result = _source_funds(
            desired=100_000,
            market_return=0.05,
            panic_threshold=-0.15,
            equity=-50_000,
            cash=20_000,
        )
        assert result == 20_000


class TestRunSimulation:
    """Test the Monte Carlo simulation."""
    
    @pytest.fixture
    def mock_residuals(self):
        """Create deterministic residuals for testing."""
        # Mean 0 residuals: [-0.1, 0, 0.1] for predictable behavior
        return np.array([-0.10, 0.0, 0.10])
    
    def test_simulation_shape(self, mock_residuals):
        """Output arrays should have correct dimensions."""
        model = RandomWalkMarket(mu=0.08, residuals=mock_residuals)
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=100,
            market_model=model
        )
        portfolio = results['portfolio_values']
        withdrawals = results['withdrawal_values']
        
        assert portfolio.shape == (11, 100)  # years+1 x n_paths
        assert withdrawals.shape == (10, 100)  # years x n_paths
    
    def test_initial_value_correct(self, mock_residuals):
        """Year 0 should equal initial net worth."""
        model = RandomWalkMarket(mu=0.08, residuals=mock_residuals)
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            market_model=model
        )
        portfolio = results['portfolio_values']
        
        assert np.all(portfolio[0, :] == 1_000_000)
    
    def test_reproducibility_with_seed(self, mock_residuals):
        """Same seed should produce same results."""
        # Set numpy seed for reproducibility
        np.random.seed(123)
        model1 = RandomWalkMarket(mu=0.08, residuals=mock_residuals)
        results1 = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=50,
            market_model=model1
        )
        portfolio1 = results1['portfolio_values']

        np.random.seed(123)
        model2 = RandomWalkMarket(mu=0.08, residuals=mock_residuals)
        results2 = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=50,
            market_model=model2
        )
        portfolio2 = results2['portfolio_values']

        np.testing.assert_array_equal(portfolio1, portfolio2)
    
    def test_high_spend_causes_depletion(self, mock_residuals):
        """Excessive spending should deplete portfolio."""
        # Use negative residuals to ensure depletion
        bad_residuals = np.array([-0.20, -0.15, -0.10])
        model = RandomWalkMarket(mu=0.0, residuals=bad_residuals)
        results = run_simulation(
            initial_net_worth=500_000,
            annual_spend=100_000,  # 20% withdrawal rate
            buffer_years=0,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.05,
            n_paths=100,
            market_model=model
        )
        portfolio = results['portfolio_values']
        # Some paths should hit zero or near-zero
        final_values = portfolio[-1, :]
        # With consistently negative returns and high spending, portfolio should decline
        assert np.median(final_values) < 500_000
    
    def test_low_spend_preserves_wealth(self, mock_residuals):
        """Conservative spending should preserve portfolio."""
        model = RandomWalkMarket(mu=0.08, residuals=mock_residuals)
        results = run_simulation(
            initial_net_worth=5_000_000,
            annual_spend=100_000,  # 2% withdrawal rate
            buffer_years=2,
            years=30,
            panic_threshold=-0.15,
            inflation_rate=0.02,
            n_paths=100,
            market_model=model
        )
        portfolio = results['portfolio_values']
        # Median should stay positive
        median_final = np.median(portfolio[-1, :])
        assert median_final > 0

    def test_negative_equity_does_not_create_funds(self):
        """Withdrawal sourcing should not increase equity when returns go below -100%."""
        residuals = np.array([-2.0])  # Forces equity wipeout
        model = RandomWalkMarket(mu=0.0, residuals=residuals)
        results = run_simulation(
            initial_net_worth=100_000,
            annual_spend=50_000,
            buffer_years=0,
            years=1,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=1,
            market_model=model
        )
        portfolio = results['portfolio_values']
        withdrawals = results['withdrawal_values']

        # Equity should be clamped to zero and withdrawals limited to available assets
        assert portfolio[-1, 0] == 0
        assert withdrawals[0, 0] == 0
    
    def test_buffer_allocation(self, mock_residuals):
        """Cash buffer should be properly allocated."""
        # With 2 years buffer at 50k/year = 100k cash
        model = RandomWalkMarket(mu=0.0, residuals=np.array([0.0]))
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=50_000,
            buffer_years=2,  # 100k in cash
            years=1,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=1,
            market_model=model
        )
        portfolio = results['portfolio_values']
        # With 0% return and 0 inflation, after withdrawal of 50k
        # from 900k equity (since 100k is cash), we should have ~850k equity + some cash
        # Year 0: 1M, Year 1: ~950k (after 50k withdrawal from equity)
        assert portfolio[1, 0] < 1_000_000


class TestCalculateStatistics:
    """Test statistical calculations."""
    
    def test_percentile_ordering(self):
        """Lower percentile should be <= median <= upper."""
        # Create test data with known distribution
        np.random.seed(42)
        portfolio = np.random.randn(31, 1000) * 100000 + 1000000
        withdrawals = np.random.randn(30, 1000) * 10000 + 50000
        
        stats = calculate_statistics(portfolio, withdrawals, confidence=0.90)
        
        # Check ordering for all years
        assert np.all(stats['portfolio']['lower'] <= stats['portfolio']['median'])
        assert np.all(stats['portfolio']['median'] <= stats['portfolio']['upper'])
    
    def test_confidence_interval_width(self):
        """Higher confidence should give wider intervals."""
        np.random.seed(42)
        portfolio = np.random.randn(31, 1000) * 100000 + 1000000
        withdrawals = np.random.randn(30, 1000) * 10000 + 50000
        
        stats_90 = calculate_statistics(portfolio, withdrawals, confidence=0.90)
        stats_95 = calculate_statistics(portfolio, withdrawals, confidence=0.95)
        
        width_90 = stats_90['portfolio']['upper'] - stats_90['portfolio']['lower']
        width_95 = stats_95['portfolio']['upper'] - stats_95['portfolio']['lower']
        
        # 95% CI should be wider than 90% CI
        assert np.all(width_95 >= width_90)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_buffer_years(self):
        """Should work with no cash buffer."""
        residuals = np.array([0.0, 0.05, -0.05])
        model = RandomWalkMarket(mu=0.08, residuals=residuals)
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=0,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            market_model=model
        )
        portfolio = results['portfolio_values']
        assert portfolio.shape == (6, 10)
        assert np.all(portfolio[0, :] == 1_000_000)


class TestMeanRevertingMarket:
    """Test the MeanRevertingMarket class for AR modeling."""

    def test_initialization(self):
        """Test basic initialization."""
        model = MeanRevertingMarket()
        assert model.ar_order == 1
        assert model.ar_coeffs is None
        assert model.intercept is None
        assert model.residual_std is None
        assert len(model.history_window) == 1

    def test_calibration(self):
        """Test model calibration with synthetic data."""
        model = MeanRevertingMarket(ar_order=1)

        # Create synthetic AR(1) data
        np.random.seed(42)
        n_points = 50  # 50 years of annual data

        # True AR(1) parameters: y_t = 0.08 + 0.3 * y_{t-1} + noise
        true_intercept = 0.08
        true_ar_coeff = 0.3
        true_std = 0.02

        # Generate synthetic AR(1) process
        data = [0.08]  # Start with long-term mean
        for _ in range(n_points - 1):
            noise = np.random.normal(0, true_std)
            next_val = true_intercept + true_ar_coeff * data[-1] + noise
            data.append(next_val)

        # Calibrate model
        model.calibrate_from_history(data)

        # Check that parameters are set
        assert model.intercept is not None
        assert model.ar_coeffs is not None
        assert model.residual_std is not None

        # Parameters should be reasonable
        assert isinstance(model.intercept, (int, float))
        assert len(model.ar_coeffs) == 1
        assert model.residual_std > 0

    def test_simulation_requires_calibration(self):
        """Test that simulation fails without calibration."""
        model = MeanRevertingMarket()

        with pytest.raises(ValueError, match="Model not calibrated"):
            model.simulate_year(np.array([0.04]), 1)

    def test_simulation_output_shape(self):
        """Test simulation output dimensions."""
        model = MeanRevertingMarket(ar_order=1)

        # Calibrate with synthetic data
        data = np.random.normal(0.08, 0.02, 30)
        model.calibrate_from_history(data)

        # Simulate - returns 1D array of returns for n_paths simulations
        returns = model.simulate_year(np.array([0.08]), 5)

        # Check shape: (simulations,)
        assert returns.shape == (5,)

    def test_simulation_positive_growth(self):
        """Test that simulated returns are reasonable."""
        model = MeanRevertingMarket(ar_order=1)

        # Calibrate with synthetic data
        data = np.random.normal(0.08, 0.02, 30)
        model.calibrate_from_history(data)

        # Simulate
        returns = model.simulate_year(np.array([0.08]), 10)

        # Returns should be finite numbers
        assert np.all(np.isfinite(returns))
        assert len(returns) == 10


class TestARSimulation:
    """Test simulation with AR model enabled."""

    @pytest.fixture
    def calibrated_ar_model(self):
        """Create a calibrated AR model."""
        model = MeanRevertingMarket(ar_order=1)
        yields = np.random.normal(0.04, 0.01, 120)
        model.calibrate_from_history(yields)
        return model

    def test_ar_simulation_shape(self, calibrated_ar_model):
        """Test that AR simulation produces correct output shape."""
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            market_model=calibrated_ar_model
        )
        portfolio = results['portfolio_values']
        withdrawals = results['withdrawal_values']

        assert portfolio.shape == (6, 10)  # years+1 x n_paths
        assert withdrawals.shape == (5, 10)  # years x n_paths

    def test_ar_initial_value(self, calibrated_ar_model):
        """Test that initial portfolio value is correct in AR mode."""
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            market_model=calibrated_ar_model
        )
        portfolio = results['portfolio_values']

        assert np.all(portfolio[0, :] == 1_000_000)

    def test_ar_vs_historical_mode(self, calibrated_ar_model):
        """Test that AR and historical modes produce different results."""
        # Set seed for reproducibility
        np.random.seed(42)

        # Run historical mode
        model_hist = RandomWalkMarket(mu=0.08, residuals=np.array([-0.1, 0.0, 0.1]))
        results_hist = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.0,  # No inflation for cleaner comparison
            n_paths=100,
            market_model=model_hist
        )
        portfolio_hist = results_hist['portfolio_values']

        # Set seed again for fair comparison
        np.random.seed(42)

        # Run AR mode
        results_ar = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=100,
            market_model=calibrated_ar_model
        )
        portfolio_ar = results_ar['portfolio_values']

        # Results should be different (though this is probabilistic)
        # We check that at least one path differs
        assert not np.allclose(portfolio_hist, portfolio_ar)
    
    def test_zero_inflation(self):
        """Should work with no inflation."""
        residuals = np.array([0.0])
        model = RandomWalkMarket(mu=0.08, residuals=residuals)
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=10,
            market_model=model
        )
        withdrawals = results['withdrawal_values']
        # With 0% inflation, real withdrawals should be constant
        # (if not capped by 4% rule)
        assert withdrawals.shape == (5, 10)
    
    def test_single_path(self):
        """Should work with just one simulation path."""
        residuals = np.array([0.05, -0.05, 0.0])
        model = RandomWalkMarket(mu=0.08, residuals=residuals)
        results = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=1,
            market_model=model
        )
        portfolio = results['portfolio_values']
        withdrawals = results['withdrawal_values']
        
        assert portfolio.shape == (11, 1)
        assert withdrawals.shape == (10, 1)


class TestDistributionAlignment:
    """Test that simulation distribution aligns with historical data for various models."""

    @pytest.fixture
    def sp500_data(self):
        """Load and prepare S&P 500 data."""
        try:
            df = pd.read_csv("s_and_p_500_with_dividends.csv", header=None, names=["Year", "Return"])
        except FileNotFoundError:
            pytest.skip("Data file not found")
            return None

        df["Return"] = df["Return"] / 100.0
        df = df.sort_values("Year")
        
        if len(df) < 60:
            pytest.skip(f"Not enough data: {len(df)} years available, need 60.")
            
        return df

    def verify_model_fit(self, model, train_data, test_data, n_years_sim=1000, n_paths_sim=100, n_backtest_paths=2000):
        """
        Helper to verify model distribution against training data and 
        backtest against test data.
        """
        # 1. Verify Generator (Simulation ~ Train)
        # Fix seed for reproducibility
        np.random.seed(42)
        sim_matrix = model.simulate_matrix(years=n_years_sim, n_paths=n_paths_sim)
        flat_sim = sim_matrix.flatten()

        train_mean = train_data.mean()
        train_std = train_data.std()
        
        sim_mean = np.mean(flat_sim)
        sim_std = np.std(flat_sim)

        # Check moments (Mean and Std Dev)
        # Using 0.025 (2.5%) tolerance
        assert abs(sim_mean - train_mean) < 0.025, \
            f"Sim Mean {sim_mean:.3f} != Train Mean {train_mean:.3f} (Diff: {sim_mean-train_mean:.3f})"
        assert abs(sim_std - train_std) < 0.025, \
            f"Sim Std {sim_std:.3f} != Train Std {train_std:.3f} (Diff: {sim_std-train_std:.3f})"

        # 2. Verify Backtest (Test ~ Simulation)
        # Simulate the specific test period length
        test_len = len(test_data)
        sim_test_period = model.simulate_matrix(years=test_len, n_paths=n_backtest_paths)
        
        # Compare Mean Return of the test period
        sim_period_means = np.mean(sim_test_period, axis=0)
        actual_period_mean = test_data.mean()

        # Check if actual is within 0.5th and 99.5th percentile (wide confidence)
        lower_bound = np.percentile(sim_period_means, 0.5)
        upper_bound = np.percentile(sim_period_means, 99.5)

        assert lower_bound < actual_period_mean < upper_bound, \
            f"Actual Mean {actual_period_mean:.3f} outside Sim bounds [{lower_bound:.3f}, {upper_bound:.3f}]"

    def test_ar1_alignment(self, sp500_data):
        """Test MeanRevertingMarket with AR(1)."""
        if sp500_data is None: return

        test_df = sp500_data.iloc[-10:]
        train_df = sp500_data.iloc[-60:-10]
        train_values = train_df["Return"].values

        model = MeanRevertingMarket(ar_order=1)
        model.calibrate_from_history(train_values)
        
        self.verify_model_fit(model, train_df["Return"], test_df["Return"])

    def test_ar2_alignment(self, sp500_data):
        """Test MeanRevertingMarket with AR(2)."""
        if sp500_data is None: return

        test_df = sp500_data.iloc[-10:]
        train_df = sp500_data.iloc[-60:-10]
        train_values = train_df["Return"].values

        model = MeanRevertingMarket(ar_order=2)
        model.calibrate_from_history(train_values)
        
        self.verify_model_fit(model, train_df["Return"], test_df["Return"])

    def test_random_walk_alignment(self, sp500_data):
        """Test RandomWalkMarket."""
        if sp500_data is None: return

        test_df = sp500_data.iloc[-10:]
        train_df = sp500_data.iloc[-60:-10]
        train_values = train_df["Return"].values

        # Manually calculate mu and residuals for initialization
        mu = np.mean(train_values)
        residuals = train_values - mu
        
        model = RandomWalkMarket(mu=mu, residuals=residuals)
        
        self.verify_model_fit(model, train_df["Return"], test_df["Return"])

    def test_block_bootstrap_alignment(self, sp500_data):
        """Test BlockBootstrapMarket."""
        if sp500_data is None: return

        test_df = sp500_data.iloc[-10:]
        train_df = sp500_data.iloc[-60:-10]
        train_values = train_df["Return"].values

        # BlockBootstrap re-samples from history directly
        model = BlockBootstrapMarket(history_returns=train_values, block_size=5)
        
        self.verify_model_fit(model, train_df["Return"], test_df["Return"])

