import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cash Buffer Impact Analysis\n",
    "\n",
    "This notebook investigates the hypothesis that **Cash Buffers harm portfolio performance** more than they help, even for retirees concerned with volatility.\n",
    "\n",
    "We will simulate different `buffer_years` (0 to 5) and compare:\n",
    "1.  **Terminal Wealth:** Does the cash drag significantly reduce final portfolio value?\n",
    "2.  **Success Rate / Survival:** Does the buffer actually prevent ruin in worst-case scenarios?\n",
    "3.  **Downside Protection:** Look at the 1st and 5th percentile outcomes.\n",
    "\n",
    "**Simulation Parameters:**\n",
    "*   Model: AR(1) (Calibrated to full history)\n",
    "*   Horizon: 30 Years\n",
    "*   Initial Net Worth: $1,000,000\n",
    "*   Annual Spend: $40,000 (4% rule)\n",
    "*   Panic Threshold: -15% (Market drop triggering cash usage)\n",
    "*   Paths: 2,000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.graph_objects as go\n",
    "import plotly.express as px\n",
    "from simulator import MeanRevertingMarket, run_simulation, get_sp500_data\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Setup Simulation\n",
    "history = get_sp500_data(history_years=60)\n",
    "model = MeanRevertingMarket(ar_order=1)\n",
    "model.calibrate_from_history(history)\n",
    "\n",
    "base_params = {\n",
    "    \"initial_net_worth\": 1_000_000,\n",
    "    \"annual_spend\": 40_000,\n",
    "    \"years\": 30,\n",
    "    \"panic_threshold\": -0.15,\n",
    "    \"inflation_rate\": 0.03,\n",
    "    \"n_paths\": 2000,\n",
    "    \"market_model\": model\n",
    "}\n",
    "\n",
    "buffer_options = [0, 1, 2, 3, 5]\n",
    "results = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Run Simulations\n",
    "np.random.seed(42) # Shared seed for fair comparison across buffers\n",
    "\n",
    "for buf in buffer_options:\n",
    "    print(f\"Simulating Buffer: {buf} years...\")\n",
    "    # We must reset seed inside the loop or pass same seed to ensure \n",
    "    # the MARKET returns are identical for every buffer scenario.\n",
    "    # However, the simulator generates market matrix internally.\n",
    "    # To strictly control this, we'd need to pass the matrix, but \n",
    "    # re-seeding right before the call works if the calls are identical order.\n",
    "    np.random.seed(42) \n",
    "    \n",
    "    sim_out = run_simulation(\n",
    "        buffer_years=buf,\n",
    "        **base_params\n",
    "    )\n",
    "    results[buf] = sim_out[\"portfolio_values\"][-1, :] # Terminal values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Analyze Results\n",
    "metrics = []\n",
    "\n",
    "for buf, terminal_values in results.items():\n",
    "    # Success: Portfolio > 0\n",
    "    success_rate = np.mean(terminal_values > 0)\n",
    "    \n",
    "    metrics.append({\n",
    "        \"Buffer Years\": str(buf),\n",
    "        \"Median Wealth\": np.median(terminal_values),\n",
    "        \"Top 25% Wealth\": np.percentile(terminal_values, 75),\n",
    "        \"Bottom 25% Wealth\": np.percentile(terminal_values, 25),\n",
    "        \"Worst 5% Wealth\": np.percentile(terminal_values, 5),\n",
    "        \"Success Rate\": success_rate\n",
    "    })\n",
    "\n",
    "metrics_df = pd.DataFrame(metrics)\n",
    "metrics_df.style.format({\n",
    "    \"Median Wealth\": \"${:,.0f}\",\n",
    "    \"Top 25% Wealth\": \"${:,.0f}\",\n",
    "    \"Bottom 25% Wealth\": \"${:,.0f}\",\n",
    "    \"Worst 5% Wealth\": \"${:,.0f}\",\n",
    "    \"Success Rate\": \"{:.1%}\"\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Visualization: Box Plots of Terminal Wealth\n",
    "# Reshape for Plotly\n",
    "plot_data = []\n",
    "for buf, values in results.items():\n",
    "    for v in values:\n",
    "        plot_data.append({\"Buffer Years\": str(buf), \"Terminal Wealth\": v})\n",
    "        \n",
    "df_plot = pd.DataFrame(plot_data)\n",
    "\n",
    "fig = px.box(\n",
    "    df_plot, \n",
    "    x=\"Buffer Years\", \n",
    "    y=\"Terminal Wealth\", \n",
    "    title=\"Distribution of Terminal Wealth (30y) by Cash Buffer Size\",\n",
    "    points=False # Hide outliers for cleaner view of mass\n",
    ")\n",
    "fig.update_layout(height=600)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Downside Focus (Tail Risk)\n",
    "# Zooming in on the worst 10% outcomes to see if buffer helps THERE.\n",
    "fig2 = px.strip(\n",
    "    df_plot[df_plot[\"Terminal Wealth\"] < df_plot[\"Terminal Wealth\"].quantile(0.2)], \n",
    "    x=\"Buffer Years\", \n",
    "    y=\"Terminal Wealth\",\n",
    "    title=\"Worst 20% of Outcomes: Does Buffer Help Survive?\"\n",
    ")\n",
    "fig2.update_layout(height=500)\n",
    "fig2.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("cash_buffer_analysis.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)
