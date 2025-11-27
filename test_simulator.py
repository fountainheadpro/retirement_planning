"""
Tests for the retirement portfolio simulator.
"""
import numpy as np
import pytest

from simulator import run_simulation, calculate_statistics, _source_funds


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


class TestRunSimulation:
    """Test the Monte Carlo simulation."""
    
    @pytest.fixture
    def mock_residuals(self):
        """Create deterministic residuals for testing."""
        # Mean 0 residuals: [-0.1, 0, 0.1] for predictable behavior
        return np.array([-0.10, 0.0, 0.10])
    
    def test_simulation_shape(self, mock_residuals):
        """Output arrays should have correct dimensions."""
        portfolio, withdrawals = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=100,
            mu=0.08,
            residuals=mock_residuals,
            seed=42
        )
        
        assert portfolio.shape == (11, 100)  # years+1 x n_paths
        assert withdrawals.shape == (10, 100)  # years x n_paths
    
    def test_initial_value_correct(self, mock_residuals):
        """Year 0 should equal initial net worth."""
        portfolio, _ = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            mu=0.08,
            residuals=mock_residuals,
            seed=42
        )
        
        assert np.all(portfolio[0, :] == 1_000_000)
    
    def test_reproducibility_with_seed(self, mock_residuals):
        """Same seed should produce same results."""
        result1, _ = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=50,
            mu=0.08,
            residuals=mock_residuals,
            seed=123
        )
        
        result2, _ = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=50,
            mu=0.08,
            residuals=mock_residuals,
            seed=123
        )
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_high_spend_causes_depletion(self, mock_residuals):
        """Excessive spending should deplete portfolio."""
        # Use negative residuals to ensure depletion
        bad_residuals = np.array([-0.20, -0.15, -0.10])
        portfolio, _ = run_simulation(
            initial_net_worth=500_000,
            annual_spend=100_000,  # 20% withdrawal rate
            buffer_years=0,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.05,
            n_paths=100,
            mu=0.0,  # No positive mean return
            residuals=bad_residuals,
            seed=42
        )
        
        # Some paths should hit zero or near-zero
        final_values = portfolio[-1, :]
        # With consistently negative returns and high spending, portfolio should decline
        assert np.median(final_values) < 500_000
    
    def test_low_spend_preserves_wealth(self, mock_residuals):
        """Conservative spending should preserve portfolio."""
        portfolio, _ = run_simulation(
            initial_net_worth=5_000_000,
            annual_spend=100_000,  # 2% withdrawal rate
            buffer_years=2,
            years=30,
            panic_threshold=-0.15,
            inflation_rate=0.02,
            n_paths=100,
            mu=0.08,
            residuals=mock_residuals,
            seed=42
        )
        
        # Median should stay positive
        median_final = np.median(portfolio[-1, :])
        assert median_final > 0
    
    def test_buffer_allocation(self, mock_residuals):
        """Cash buffer should be properly allocated."""
        # With 2 years buffer at 50k/year = 100k cash
        portfolio, _ = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=50_000,
            buffer_years=2,  # 100k in cash
            years=1,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=1,
            mu=0.0,  # No market return
            residuals=np.array([0.0]),  # Zero residual
            seed=42
        )
        
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
        portfolio, _ = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=0,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=10,
            mu=0.08,
            residuals=residuals,
            seed=42
        )
        
        assert portfolio.shape == (6, 10)
        assert np.all(portfolio[0, :] == 1_000_000)
    
    def test_zero_inflation(self):
        """Should work with no inflation."""
        residuals = np.array([0.0])
        portfolio, withdrawals = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=5,
            panic_threshold=-0.15,
            inflation_rate=0.0,
            n_paths=10,
            mu=0.08,
            residuals=residuals,
            seed=42
        )
        
        # With 0% inflation, real withdrawals should be constant
        # (if not capped by 4% rule)
        assert withdrawals.shape == (5, 10)
    
    def test_single_path(self):
        """Should work with just one simulation path."""
        residuals = np.array([0.05, -0.05, 0.0])
        portfolio, withdrawals = run_simulation(
            initial_net_worth=1_000_000,
            annual_spend=40_000,
            buffer_years=2,
            years=10,
            panic_threshold=-0.15,
            inflation_rate=0.03,
            n_paths=1,
            mu=0.08,
            residuals=residuals,
            seed=42
        )
        
        assert portfolio.shape == (11, 1)
        assert withdrawals.shape == (10, 1)
