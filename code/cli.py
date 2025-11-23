"""CLI interface for weather prediction and simulation"""

import click
from datetime import datetime
from markov_weather.predictor import Predictor
from markov_weather.model import WeatherMarkovChain
from markov_weather.data_loader import WeatherDataLoader
from markov_weather.matrix_builder import MatrixBuilder
from markov_weather.matrix_builder_2order import HigherOrderMatrixBuilder
from markov_weather.simulator import WeatherSimulator
from markov_weather.validation import ValidationAnalyzer
from markov_weather.model_2order import HigherOrderMarkovChain
from markov_weather.validation_2order import HigherOrderValidationAnalyzer


@click.group()
def cli():
    """Markov Chain Weather Prediction CLI"""
    pass


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--state', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Current weather state')
@click.option('--days', type=int, default=3, help='Number of forecast days')
@click.option('--month', help='Month (e.g. january). If not specified: current month')
@click.option('--probabilities', is_flag=True, help='Show probabilities instead of only most likely states')
def predict(region: str, state: str, days: int, month: str, probabilities: bool):
    """Make a weather prediction"""

    # If month not specified, use current month
    if not month:
        month = datetime.now().strftime('%B').lower()

    try:
        predictor = Predictor(region)

        if probabilities:
            # Show probabilities
            predictions = predictor.predict(state, days, month)
            click.echo(f"\n📊 Weather forecast for {region} ({month})")
            click.echo(f"Current weather: {state}\n")

            for day_num, probs in enumerate(predictions, 1):
                click.echo(f"Day {day_num}:")
                for weather_state in WeatherMarkovChain.STATES:
                    prob_percent = probs[weather_state] * 100
                    bar_length = int(prob_percent / 5)
                    bar = "█" * bar_length
                    click.echo(f"  {weather_state:15} {prob_percent:5.1f}% {bar}")
                click.echo()

        else:
            # Show only most likely states
            most_likely = predictor.predict_most_likely(state, days, month)
            click.echo(f"\n☀️  Weather forecast for {region} ({month})")
            click.echo(f"Current weather: {state}\n")

            for day_num, weather_state in enumerate(most_likely, 1):
                click.echo(f"Day {day_num}: {weather_state}")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("Note: Matrices must be generated first with 'python cli.py generate-matrices'", err=True)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--data-file', default='data/raw/sample_weather.csv', help='Path to CSV file with weather data')
def generate_matrices(region: str, data_file: str):
    """Generate transition matrices (1st order) from weather data"""

    try:
        builder = MatrixBuilder(region)
        builder.build_from_file(data_file)
        click.echo(f"✅ Matrices for {region} successfully generated!")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("Note: First generate sample data with 'python cli.py generate-sample-data'", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--data-file', default='data/raw/sample_weather.csv', help='Path to CSV file with weather data')
@click.option('--order', type=int, default=2, help='Markov chain order (2 or 3)')
def generate_matrices_higher(region: str, data_file: str, order: int):
    """Generate higher-order transition matrices (2nd or 3rd order) from weather data"""

    if order not in [2, 3]:
        click.echo(f"❌ Error: Order must be 2 or 3, not {order}", err=True)
        return

    try:
        builder = HigherOrderMatrixBuilder(region, order=order)
        click.echo(f"\n🔄 Generating {order}. order matrices for {region}...")
        builder.print_matrix_info()
        click.echo()
        builder.build_from_file(data_file)
        click.echo(f"\n✅ Matrices ({order}. order) for {region} successfully generated!")
        click.echo(f"   Output directory: data/matrices/order{order}/{region}/")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo("Note: First generate sample data with 'python cli.py generate-sample-data'", err=True)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--output-file', default='data/raw/sample_weather.csv', help='Output file for sample data')
@click.option('--records', type=int, default=1095, help='Number of records (3 years = ~1095 days)')
def generate_sample_data(output_file: str, records: int):
    """Generate realistic sample data"""

    loader = WeatherDataLoader()
    data = loader.create_sample_data(output_file, records)

    click.echo(f"✅ Sample data generated: {output_file}")
    click.echo(f"   Records: {len(data)}")
    click.echo(f"   Period: {data['date'].min()} to {data['date'].max()}")
    click.echo(f"\n   Weather state distribution:")
    for state, count in data['weather_state'].value_counts().items():
        percent = (count / len(data)) * 100
        click.echo(f"   - {state:15}: {count:4} days ({percent:5.1f}%)")


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--month', help='Month (e.g. january)')
def show_matrix(region: str, month: str):
    """Show a transition matrix"""

    if not month:
        month = datetime.now().strftime('%B').lower()

    try:
        chain = WeatherMarkovChain(region, month)

        click.echo(f"\n📋 Transition matrix for {region} ({month})\n")
        click.echo("From / To:       " + "  ".join(f"{s:8}" for s in WeatherMarkovChain.STATES))
        click.echo("-" * 80)

        for from_state in WeatherMarkovChain.STATES:
            probs = chain.predict_next_day(from_state)
            row = f"{from_state:15} "
            for to_state in WeatherMarkovChain.STATES:
                prob = probs[to_state]
                row += f"  {prob*100:5.0f}%"
            click.echo(row)

        # Validation
        is_valid = WeatherMarkovChain.verify_matrix_validity(chain.transition_matrix)
        click.echo()
        click.echo(f"✓ Matrix valid: {is_valid}")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--state', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Starting state')
@click.option('--days', type=int, default=30, help='Number of days to simulate')
@click.option('--month', help='Month (e.g. january). If not specified: current month')
@click.option('--runs', type=int, default=1000, help='Number of simulations')
@click.option('--show-sequences', is_flag=True, help='Show first 5 sequences')
def simulate(region: str, state: str, days: int, month: str, runs: int, show_sequences: bool):
    """Simulate weather sequences stochastically"""

    if not month:
        month = datetime.now().strftime('%B').lower()

    try:
        simulator = WeatherSimulator(region)

        click.echo(f"\n🎲 Simulating {runs} weather sequences for {region}")
        click.echo(f"   Start: {state}, Duration: {days} days, Month: {month}\n")

        sequences = simulator.simulate_many(state, days, month, runs)
        stats = simulator.get_sequence_statistics(sequences)

        # Show some sequences as examples
        if show_sequences:
            click.echo("📋 Example sequences:")
            for i, seq in enumerate(sequences[:5], 1):
                seq_str = " → ".join(seq[:15]) + ("..." if len(seq) > 15 else "")
                click.echo(f"   {i}. {seq_str}")
            click.echo()

        # Statistics
        click.echo("📊 Weather state frequencies:")
        for state_name in WeatherMarkovChain.STATES:
            if state_name in stats['state_probabilities']:
                prob = stats['state_probabilities'][state_name] * 100
                count = stats['state_counts'][state_name]
                bar = "█" * int(prob / 5)
                click.echo(f"  {state_name:15}: {prob:5.1f}% ({count:6} / {stats['total_sequences'] * days}) {bar}")

        # Longest runs
        click.echo("\n🔗 Longest runs (per state):")
        for state_name in WeatherMarkovChain.STATES:
            if state_name in stats['consecutive_runs'] and stats['consecutive_runs'][state_name]:
                runs_list = stats['consecutive_runs'][state_name]
                max_run = max(runs_list)
                avg_run = sum(runs_list) / len(runs_list)
                click.echo(f"  {state_name:15}: max={max_run:2} days, avg={avg_run:.1f} days")

        # Top transitions
        click.echo("\n→ Top 5 transitions:")
        top_transitions = sorted(
            stats['transition_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for transition, count in top_transitions:
            prob = (count / (stats['total_sequences'] * (days - 1))) * 100
            click.echo(f"  {transition:30}: {prob:5.1f}%")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--state', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Starting state')
@click.option('--target', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Target state')
@click.option('--days', type=int, default=30, help='Time window in days')
@click.option('--month', help='Month (e.g. january). If not specified: current month')
@click.option('--runs', type=int, default=10000, help='Number of Monte Carlo simulations')
def probability(region: str, state: str, target: str, days: int, month: str, runs: int):
    """Calculate probability of a transition using Monte Carlo method"""

    if not month:
        month = datetime.now().strftime('%B').lower()

    try:
        simulator = WeatherSimulator(region)

        click.echo(f"\n🎲 Calculating probability with {runs} Monte Carlo simulations")
        click.echo(f"   Start: {state} → Target: {target}")
        click.echo(f"   Time window: {days} days, Month: {month}\n")

        prob = WeatherSimulator.monte_carlo_probability(
            simulator, state, target, days, month, runs
        )

        click.echo(f"📊 Probability that {target} occurs within the next {days} days:")
        click.echo(f"   {prob:.2f}%\n")

        # Interpretations
        if prob > 80:
            click.echo("   ✓ Very likely!")
        elif prob > 50:
            click.echo("   ◐ Likely")
        elif prob > 20:
            click.echo("   ◑ Possible")
        else:
            click.echo("   ✗ Unlikely")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--state', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Starting state')
@click.option('--days', type=int, default=30, help='Number of days to simulate')
@click.option('--months', multiple=True, required=True, help='Months to compare (e.g. --months january --months july)')
@click.option('--runs', type=int, default=1000, help='Simulations per month')
def compare_months(region: str, state: str, days: int, months: tuple, runs: int):
    """Compare weather patterns across multiple months"""

    if not months:
        click.echo("❌ At least two months required!", err=True)
        return

    try:
        simulator = WeatherSimulator(region)

        click.echo(f"\n📊 Comparing weather patterns across {len(months)} months")
        click.echo(f"   Start: {state}, {days} days, {runs} simulations per month\n")

        sequences_by_month = {}
        for month in months:
            sequences_by_month[month] = simulator.simulate_many(state, days, month, runs)

        comparison = WeatherSimulator.compare_month_statistics(sequences_by_month)

        # Table: State frequencies per month
        click.echo("📈 State frequencies by month:")
        click.echo("-" * 80)

        header = "State".ljust(15) + "".join(f"{m:>12}" for m in months)
        click.echo(header)
        click.echo("-" * 80)

        for weather_state in WeatherMarkovChain.STATES:
            row = weather_state.ljust(15)
            for month in months:
                prob = comparison[month]['state_probabilities'].get(weather_state, 0)
                row += f"{prob*100:>11.1f}%"
            click.echo(row)

        click.echo("-" * 80)

        # Longest runs per month
        click.echo("\n🔗 Longest runs per month:")
        for weather_state in WeatherMarkovChain.STATES:
            click.echo(f"\n  {weather_state}:")
            for month in months:
                runs_list = comparison[month]['consecutive_runs'].get(weather_state, [])
                if runs_list:
                    max_run = max(runs_list)
                    avg_run = sum(runs_list) / len(runs_list)
                    click.echo(f"    {month:12}: max={max_run:2} days, avg={avg_run:.1f} days")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--state', type=click.Choice(WeatherMarkovChain.STATES), required=True, help='Starting state')
@click.option('--days', type=int, default=30, help='Number of days to simulate')
@click.option('--month', help='Month')
@click.option('--event', required=True, help='Event (e.g. "rainy", "sunny→cloudy")')
@click.option('--min-length', type=int, default=3, help='Minimum run length')
@click.option('--runs', type=int, default=10000, help='Number of simulations')
def rare_event(region: str, state: str, days: int, month: str, event: str, min_length: int, runs: int):
    """Find how often rare events occur"""

    if not month:
        month = datetime.now().strftime('%B').lower()

    try:
        simulator = WeatherSimulator(region)

        click.echo(f"\n🔍 Searching for rare events")
        click.echo(f"   Event: {event}")
        if "→" not in event:
            click.echo(f"   Minimum run length: {min_length} days")
        click.echo(f"   Start: {state}, {days} days, Month: {month}")
        click.echo(f"   Simulations: {runs}\n")

        sequences = simulator.simulate_many(state, days, month, runs)
        prob, count = WeatherSimulator.find_rare_events(sequences, event, min_length)

        click.echo(f"📊 Results:")
        click.echo(f"   Event occurs in: {prob:.2f}% of sequences ({count} / {runs})")

        if prob < 1:
            click.echo(f"   🔴 Very rare!")
        elif prob < 5:
            click.echo(f"   🟠 Rare")
        elif prob < 20:
            click.echo(f"   🟡 Occasional")
        else:
            click.echo(f"   🟢 Frequent")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='nordfriesland', help='Region name')
@click.option('--data-file', default='data/raw/nordfriesland_2024.csv', help='CSV file with real data')
def validate(region: str, data_file: str):
    """Validate the model against real weather data"""

    try:
        click.echo(f"\n🔍 Validating model for {region}")
        click.echo(f"   Data: {data_file}\n")

        validator = ValidationAnalyzer(region, data_file)

        # 1. State distribution
        click.echo("📊 1. Comparison: State frequencies (real data vs. simulation)")
        click.echo("=" * 100)
        dist_comparison = validator.compare_state_distribution()

        # Print table
        print(f"\n{'Month':<12} {'State':<15} {'Real %':<12} {'Sim. %':<12} {'Deviation':<12}")
        print("-" * 100)

        for month in sorted(dist_comparison.keys()):
            states = dist_comparison[month]
            month_printed = False
            for state in WeatherMarkovChain.STATES:
                if state in states:
                    data = states[state]
                    month_str = month if not month_printed else ""
                    diff_indicator = "✓" if data['diff'] < 5 else "◐" if data['diff'] < 10 else "✗"
                    print(
                        f"{month_str:<12} {state:<15} {data['real']:>6.1f}%  "
                        f"{data['simulated']:>6.1f}%  {data['diff']:>6.1f}% {diff_indicator}"
                    )
                    month_printed = True

        # Calculate MAE
        overall_mae, month_errors = validator.calculate_mae(dist_comparison)
        click.echo("\n" + "=" * 100)
        click.echo(f"\n📈 MAE (Mean Absolute Error) per month:")
        click.echo("-" * 100)
        for month in sorted(month_errors.keys()):
            mae = month_errors[month]
            quality = "🟢" if mae < 5 else "🟡" if mae < 10 else "🔴"
            print(f"  {month:12} : {mae:5.2f}% {quality}")

        click.echo(f"\n{'OVERALL MAE':>12} : {overall_mae:5.2f}%")

        # 2. Transitions
        click.echo("\n\n🔗 2. Comparison: Transition patterns")
        click.echo("=" * 100)

        trans_comparison = validator.compare_transitions()
        best, worst = validator.get_best_worst_transitions(trans_comparison, n=5)

        click.echo("\n✓ BEST predicted transitions:")
        print(f"{'Transition':<25} {'Real %':<12} {'Sim. %':<12} {'Deviation':<12}")
        print("-" * 100)
        for transition, data in best:
            print(
                f"{transition:<25} {data['real']:>6.1f}%  "
                f"{data['simulated']:>6.1f}%  {data['diff']:>6.1f}%"
            )

        click.echo("\n✗ WORST predicted transitions:")
        print(f"{'Transition':<25} {'Real %':<12} {'Sim. %':<12} {'Deviation':<12}")
        print("-" * 100)
        for transition, data in worst:
            print(
                f"{transition:<25} {data['real']:>6.1f}%  "
                f"{data['simulated']:>6.1f}%  {data['diff']:>6.1f}%"
            )

        # 3. Prediction accuracy
        click.echo("\n\n🎯 3. Prediction accuracy (against real data)")
        click.echo("=" * 100)

        prediction_results = validator.predict_vs_actual(days=7)

        click.echo(f"\n1-day prediction:")
        click.echo(f"  Correct:  {prediction_results['correct_next_day']} / {prediction_results['correct_next_day'] + prediction_results['incorrect_next_day']}")
        click.echo(f"  Accuracy: {prediction_results['accuracy_next_day']:.1f}%")

        click.echo(f"\n7-day prediction (correct sequences):")
        click.echo(f"  Correct:  {prediction_results['correct_sequences']} / {prediction_results['total_sequences']}")
        click.echo(f"  Accuracy: {prediction_results['sequence_accuracy']:.1f}%")

        click.echo(f"\nAccuracy per starting state:")
        print(f"{'State':<15} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
        print("-" * 100)
        for state in sorted(prediction_results['predictions_by_state'].keys()):
            stats = prediction_results['predictions_by_state'][state]
            if stats['total'] > 0:
                print(
                    f"{state:<15} {stats['correct']:<8} {stats['total']:<8} "
                    f"{stats['accuracy']:>6.1f}%"
                )

        # Overall score
        click.echo("\n\n⭐ OVERALL RATING")
        click.echo("=" * 100)
        confidence = validator.get_confidence_score()

        if confidence >= 80:
            rating = "🟢 Excellent"
        elif confidence >= 70:
            rating = "🟡 Good"
        elif confidence >= 60:
            rating = "🟠 Satisfactory"
        else:
            rating = "🔴 Insufficient"

        click.echo(f"\nConfidence score: {confidence:.1f}% {rating}\n")

        click.echo("Interpretation:")
        click.echo("  • < 60%: Model does not fit real data well")
        click.echo("  • 60-70%: Model has recognized basic patterns")
        click.echo("  • 70-80%: Model is reliable")
        click.echo("  • > 80%: Model is highly reliable")

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option('--region', default='schleswig_holstein', help='Region name')
@click.option('--data-file', default='data/raw/sample_weather.csv', help='Path to CSV file with weather data')
@click.option('--order', type=int, default=2, help='Markov chain order (2 or 3)')
def validate_higher(region: str, data_file: str, order: int):
    """Validate a higher-order Markov model against real data"""

    if order not in [2, 3]:
        click.echo(f"❌ Error: Order must be 2 or 3, not {order}", err=True)
        return

    try:
        analyzer = HigherOrderValidationAnalyzer(region, data_file, order=order)
        analyzer.print_validation_report()

    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo(f"Note: Matrices must be generated first with 'python cli.py generate-matrices-higher'", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    cli()
