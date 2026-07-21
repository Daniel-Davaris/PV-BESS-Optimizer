def export_results(self):
    """Package window optimization results.

    Top-level dispatch:
      run_multiple_optimisations == False  →  full diagnostic export
      run_multiple_optimisations == True   →  lean export for parallel/batch runs

    Within the full export, revenue semantics depend on the optimisation directive
    and predicted_price_type:
      directive 1  (perfect foresight)     : LP used actual prices → actual_revenue == objective
      directive 2, predicted_price_type 1  : predicted prices only
      directive 2, predicted_price_type 2  : BESS instructions only (LP price = 0)
      directive 2, predicted_price_type 3  : predicted prices + BESS instructions
    """

    # ------------------------------------------------------------------ #
    # Shared helper                                                        #
    # ------------------------------------------------------------------ #

    def _compute_actual_revenue():
        """Energy trading P&L at actual spot prices: export revenue + LGC - import costs.
        Does not include demand charges or LP penalties — those are added separately in
        _add_revenue_columns so the final actual_revenue_row_total is total realised P&L."""
        n = len(self.dataset_window_subset)
        actual_price   = self.dataset_window_subset["Price"].to_numpy()
        lgc            = self.dataset_window_subset["LGC"].to_numpy()
        import_charge  = self.dataset_window_subset["Import_charge"].to_numpy()
        hte2grid       = self.dataset_window_subset["hte2grid"].to_numpy()

        pv2grid_v      = np.array([value(v) or 0.0 for v in self.pv2grid[:n]])
        dischargeE_v   = np.array([value(v) or 0.0 for v in self.dischargeE[:n]])
        grid2battery_v = np.array([value(v) or 0.0 for v in self.grid2battery[:n]])
        dischargeE_eff = dischargeE_v * hte2grid

        return (
            (pv2grid_v + dischargeE_eff) * actual_price
            + (pv2grid_v + dischargeE_eff - grid2battery_v * variables.grid_import_penalty) * lgc
            - grid2battery_v * (actual_price + import_charge)
        )

    # ------------------------------------------------------------------ #
    # Full export — single optimisation                                   #
    # ------------------------------------------------------------------ #

    def _export_single():
        """Full diagnostic DataFrame for a single (non-parallel) run.
        Contains: input columns, decision variables, objective components,
        constraint diagnostics (value / slack / dual), and revenue totals."""

        base_df = self.dataset_window_subset.copy().reset_index(drop=True)
        base_df.columns = [f"input_{col}" for col in base_df.columns]
        n = len(base_df)

        # ---- variable name parsers ---- #

        def _split_interval_name(name):
            m = re.match(r"^(.*)_(\d+)$", name)
            if not m:
                return name, None
            base, suffix = m.group(1), int(m.group(2))
            if 0 <= suffix < n and not re.search(r"_\d{4}$", base):
                return base, suffix
            return name, None

        def _split_monthly_name(name):
            m = re.match(r"^(.*)_(\d{4})_(\d{2})$", name)
            if not m:
                return name, None
            return m.group(1), (int(m.group(2)), int(m.group(3)))

        def _month_rows(ym):
            return [idx for idx, row_ym in enumerate(self.window_year_month) if row_ym == ym]

        cols = {}

        # ---- decision variables ---- #

        def _extract_decision_variables():
            for var in self.Lp.variables():
                base, i = _split_interval_name(var.name)
                if i is not None:
                    cols.setdefault(f"decision_variable_{base}", [None] * n)[i] = value(var)
                    continue

                base, ym = _split_monthly_name(var.name)
                if ym is not None:
                    col = f"decision_variable_{base}"
                    cols.setdefault(col, [None] * n)
                    for row_i in _month_rows(ym):
                        cols[col][row_i] = value(var)
                    continue

                cols[f"scalar_decision_variable_{var.name}"] = [value(var)] * n

        # ---- objective components ---- #

        def _extract_objective_components():
            for component_name, interval_exprs in self.objective_component_map.items():
                col = f"objective_component_{component_name}"
                cols[col] = [0.0] * n
                for i, expr in interval_exprs.items():
                    if i is not None and 0 <= i < n:
                        cols[col][i] += value(expr)

        # ---- constraint diagnostics ---- #
        # value = evaluated constraint expression after solve
        # slack = unused room in the constraint ("how much spare room?")
        # dual  = shadow price / marginal value of relaxing that constraint

        def _extract_constraint_diagnostics():
            for cname, c in self.Lp.constraints.items():
                i = self.constraint_map.get(cname)
                if i is None:
                    cols[f"scalar_constraint_value_{cname}"] = [value(c)] * n
                    cols[f"scalar_constraint_slack_{cname}"] = [c.slack] * n
                    cols[f"scalar_constraint_dual_{cname}"] = [c.pi] * n
                    continue

                base = cname.rsplit("_", 1)[0]
                cols.setdefault(f"constraint_value_{base}", [None] * n)[i] = value(c)
                cols.setdefault(f"constraint_slack_{base}", [None] * n)[i] = c.slack
                cols.setdefault(f"constraint_dual_{base}", [None] * n)[i] = c.pi

        _extract_decision_variables()
        _extract_objective_components()
        _extract_constraint_diagnostics()

        extra_df = pd.DataFrame(cols)

        component_cols = [c for c in extra_df.columns if c.startswith("objective_component_")]
        extra_df["objective_row_total"] = extra_df[component_cols].sum(axis=1) if component_cols else 0.0

        # ---- revenue columns ---- #
        # actual_revenue_row_total = energy P&L at actual spot prices + demand charges.
        # This is total realised P&L and is directly comparable across both directives.
        #
        # For directive 1 (perfect foresight) this equals objective_row_total exactly,
        # because the LP optimised at actual prices and curtailment_penalty / signal
        # penalties are both zero.
        #
        # For directive 2 the LP optimised at a price proxy, so objective_row_total
        # (LP objective at predicted prices) differs from actual_revenue_row_total
        # (real-world cash outcome at actual spot prices). The gap between the two
        # measures how much value was lost/gained due to imperfect price prediction.
        #
        # Demand charges (network tariffs) are included via
        # objective_component_monthly_import_export_penalties, which holds the solved
        # LP demand cost allocated to the first interval of each month.

        def _add_revenue_columns():
            actual_rev = _compute_actual_revenue()

            # Add demand charges so actual_revenue_row_total is total P&L, not just
            # spot-market P&L. The demand charge column is negative (it's a cost).
            demand_col = "objective_component_monthly_import_export_penalties"
            if demand_col in extra_df.columns:
                actual_rev = actual_rev + extra_df[demand_col].fillna(0).to_numpy()

            if variables.optimisation_directive == 2 and variables.predicted_price_type == 2:
                # BESS instructions only: LP price was 0 so the LP objective captures
                # signal penalties only — rename to avoid confusing it with cash revenue
                extra_df.rename(
                    columns={"objective_row_total": "signal_penalty_row_total"},
                    inplace=True,
                )

            extra_df["actual_revenue_row_total"] = actual_rev

        _add_revenue_columns()

        return pd.concat([base_df, extra_df], axis=1).copy()

    # ------------------------------------------------------------------ #
    # Lean export — multiple (parallel) optimisations                     #
    # ------------------------------------------------------------------ #

    def _export_multiple():
        """Minimal DataFrame for parallel/batch runs.
        Provides only what the window scheduler and downstream aggregation need:
          decision_variable_stored_charge  — SoC hand-off between windows
          objective_row_total              — LP objective per interval
          actual_revenue_row_total         — real-world revenue per interval (spot P&L + demand charges)
        """
        n = len(self.dataset_window_subset)
        stored_charge = [value(v) for v in self.stored_charge]

        objective_row_total = [0.0] * n
        demand_charges = [0.0] * n
        for component_name, interval_exprs in self.objective_component_map.items():
            for i, expr in interval_exprs.items():
                if i is not None and 0 <= i < n:
                    v = value(expr)
                    objective_row_total[i] += v
                    if component_name == "monthly_import_export_penalties":
                        demand_charges[i] += v

        actual_rev = _compute_actual_revenue() + np.array(demand_charges)

        return pd.DataFrame({
            "decision_variable_stored_charge": stored_charge,
            "objective_row_total": objective_row_total,
            "actual_revenue_row_total": actual_rev,
        })

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    if variables.run_multiple_optimisations:
        result = _export_multiple()
    else:
        result = _export_single()

    self.optimization_window_results = result
  
    return result


def export_combination_summary(df, price_col, bess_duration_h):
    """Aggregate a completed combination's per-interval DataFrame into a single summary row."""
    return pd.DataFrame([{
        "price_col": price_col,
        "bess_duration_h": bess_duration_h,
        "sum_objective_row_total": round(df["objective_row_total"].sum(), 4),
        "sum_actual_revenue_row_total": round(df["actual_revenue_row_total"].sum(), 4),
    }])
