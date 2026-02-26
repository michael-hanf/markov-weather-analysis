"""Anderson-Goodman Chi-Quadrat-Test fuer die Markov-Eigenschaft"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional

STATES = ['sunny', 'partly_cloudy', 'cloudy', 'rainy', 'stormy']
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N = len(STATES)


class MarkovPropertyTest:
    """
    Anderson-Goodman χ²-Test: Prueft ob die Markov-Eigenschaft (Gedaechtnislosigkeit) gilt.

    H₀: P(X_{t+1} | X_t, X_{t-1}) = P(X_{t+1} | X_t)
        Der morgige Zustand haengt nur vom heutigen ab, nicht vom gestrigen.

    H₁: Kenntnis des gestrigen Zustands veraendert die Vorhersage signifikant.

    Methode: Fuer jeden mittleren Zustand j wird eine s×s-Kontingenztafel
    (Vorgaengerzustand i × Nachfolgezustand k) auf Zeilenhomogenitaet getestet.
    Erwartungswert unter H₀: E[n_{ijk}] = n_{ij} * n_{jk} / n_j

    Referenz: Anderson & Goodman (1957), "Statistical Inference about Markov Chains"
    """

    def __init__(self, data_file: str, month: Optional[str] = None):
        """
        Args:
            data_file: Pfad zur CSV-Datei mit Spalte 'weather_state'
            month: Optionaler Monatsfilter (z.B. 'january')
        """
        self.data_file = data_file
        self.month_filter = month
        self.sequence = self._load_sequence()

    def _load_sequence(self) -> List[str]:
        """Laedt und filtert die Wettersequenz aus der CSV."""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        if 'weather_state' not in df.columns:
            raise ValueError(f"CSV '{self.data_file}' benoetigt Spalte 'weather_state'")

        if self.month_filter:
            df['month_name'] = df['date'].dt.strftime('%B').str.lower()
            df = df[df['month_name'] == self.month_filter]
            if df.empty:
                raise ValueError(f"Keine Daten fuer Monat '{self.month_filter}'")

        return [s for s in df['weather_state'].tolist() if s in STATE_IDX]

    def _count_ngrams(self):
        """Zaehlt Triplets n[i,j,k], Bigrams n[i,j] und Marginals n[j]."""
        seq = self.sequence
        triplets = np.zeros((N, N, N), dtype=float)
        bigrams = np.zeros((N, N), dtype=float)
        marginals = np.zeros(N, dtype=float)

        for t in range(len(seq) - 2):
            i = STATE_IDX[seq[t]]
            j = STATE_IDX[seq[t + 1]]
            k = STATE_IDX[seq[t + 2]]
            triplets[i, j, k] += 1

        for t in range(len(seq) - 1):
            i = STATE_IDX[seq[t]]
            j = STATE_IDX[seq[t + 1]]
            bigrams[i, j] += 1

        for t in range(len(seq)):
            j = STATE_IDX[seq[t]]
            marginals[j] += 1

        return triplets, bigrams, marginals

    def run(self, alpha: float = 0.05) -> Dict:
        """
        Fuehrt den Anderson-Goodman χ²-Test durch.

        Args:
            alpha: Signifikanzniveau (Standard: 0.05)

        Returns:
            Dictionary mit Gesamtergebnis und Ergebnissen pro Zustand
        """
        triplets, bigrams, marginals = self._count_ngrams()

        chi2_total = 0.0
        df_total = 0
        sparse_total = 0
        cells_total = 0
        per_state = {}

        for j in range(N):
            n_j = marginals[j]
            if n_j < 10:
                # Zu wenige Beobachtungen fuer diesen Zustand
                continue

            # Aktive Vorgaengerzustaende (Zeilen mit Daten)
            active_rows = [i for i in range(N) if bigrams[i, j] > 0]
            if len(active_rows) < 2:
                # Nur eine Vorgaengerquelle → kein Vergleich moeglich
                continue

            chi2_j = 0.0
            sparse_j = 0
            cells_j = 0

            for i in active_rows:
                n_ij = bigrams[i, j]
                for k in range(N):
                    n_ijk = triplets[i, j, k]
                    n_jk = bigrams[j, k]
                    expected = (n_ij * n_jk) / n_j

                    cells_j += 1
                    cells_total += 1

                    if expected < 5:
                        sparse_j += 1
                        sparse_total += 1
                        # Spärliche Zellen beeinflussen die χ²-Approximation;
                        # wir schließen sie aus dem Teststatistik aus.
                        continue

                    chi2_j += (n_ijk - expected) ** 2 / expected

            # Freiheitsgrade: (Anzahl aktiver Zeilen - 1) * (s - 1)
            df_j = (len(active_rows) - 1) * (N - 1)
            if df_j <= 0:
                continue

            p_j = float(1 - stats.chi2.cdf(chi2_j, df=df_j))

            per_state[STATES[j]] = {
                'chi2': round(chi2_j, 4),
                'df': df_j,
                'p_value': round(p_j, 6),
                'n_obs': int(n_j),
                'active_predecessors': len(active_rows),
                'sparse_cells': sparse_j,
                'total_cells': cells_j,
                'reject_h0': p_j < alpha,
            }

            chi2_total += chi2_j
            df_total += df_j

        p_total = float(1 - stats.chi2.cdf(chi2_total, df=df_total)) if df_total > 0 else None

        return {
            'chi2': round(chi2_total, 4),
            'df': df_total,
            'p_value': round(p_total, 6) if p_total is not None else None,
            'reject_h0': (p_total < alpha) if p_total is not None else None,
            'alpha': alpha,
            'n_observations': len(self.sequence),
            'month_filter': self.month_filter,
            'sparse_cells': sparse_total,
            'total_cells': cells_total,
            'per_state': per_state,
        }

    @staticmethod
    def print_report(result: Dict, data_file: str):
        """Gibt einen lesbaren Ergebnisbericht aus."""
        print("\n" + "=" * 70)
        print("  Anderson-Goodman χ²-Test: Markov-Eigenschaft")
        print("=" * 70)
        print(f"  Datei       : {data_file}")
        print(f"  Monat       : {result['month_filter'] or 'alle Monate'}")
        print(f"  Beobachtungen: {result['n_observations']:,}")
        print(f"  Signifikanzniveau (α): {result['alpha']}")
        print()

        # Gesamtergebnis
        print("  GESAMTTEST")
        print("  " + "-" * 50)
        print(f"  χ² = {result['chi2']:.4f},  df = {result['df']},  p = {result['p_value']:.6f}")
        print()

        if result['reject_h0'] is None:
            print("  ⚠  Zu wenige Daten fuer den Test.")
        elif result['reject_h0']:
            print("  ✗  H₀ abgelehnt: 2. Ordnung erkennbar (Gedaechtnislosigkeit verletzt)")
            print("     Das Modell profitiert moeglicherweise von einem 2nd-Order-Ansatz.")
        else:
            print("  ✓  H₀ beibehalten: Markov-Eigenschaft 1. Ordnung nicht widerlegt")
            print("     Das 1st-Order-Modell ist strukturell begruendet.")

        # Sparse-Cell-Warnung
        if result['sparse_cells'] > 0:
            pct = 100 * result['sparse_cells'] / result['total_cells']
            print(f"\n  ⚠  Sparse Cells: {result['sparse_cells']}/{result['total_cells']}"
                  f" ({pct:.0f}%) mit Erwartungswert < 5 wurden ausgeschlossen.")

        # Ergebnisse pro Zustand
        print("\n  ERGEBNISSE PRO ZUSTAND (j = Gegenwartszustand)")
        print("  " + "-" * 70)
        header = f"  {'Zustand':<15} {'n_obs':>7} {'Vorg.':>5} {'χ²':>9} {'df':>4} {'p-Wert':>10}  {'H₀'}"
        print(header)
        print("  " + "-" * 70)

        for state in STATES:
            if state not in result['per_state']:
                print(f"  {state:<15} {'–':>7} {'–':>5} {'–':>9} {'–':>4} {'–':>10}  (keine Daten)")
                continue
            r = result['per_state'][state]
            symbol = "✗ ablehnen" if r['reject_h0'] else "✓ beibehalten"
            print(
                f"  {state:<15} {r['n_obs']:>7,} {r['active_predecessors']:>5} "
                f"{r['chi2']:>9.4f} {r['df']:>4} {r['p_value']:>10.6f}  {symbol}"
            )

        print("=" * 70)
        print()
        print("  Interpretation:")
        print("  • p < α  →  H₀ ablehnen: Vergangenheit traegt zur Vorhersage bei")
        print("  • p ≥ α  →  H₀ beibehalten: 1st-Order-Annahme nicht widerlegbar")
        print("  • 'Vorg.' = Anzahl Vorgaengerzustaende mit ausreichend Daten")
        print()
