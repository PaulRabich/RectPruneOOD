from pandas import DataFrame
from tqdm import tqdm

from rect_prune_ood.lib.eval.metrics import compute_traditional_ood
from rect_prune_ood.lib.experiments.run_experiment import run_single_experiment

columns = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']


def build_table(df):
	df['FPR'] = df['FPR'] * 100
	df['AUROC'] = df['AUROC'] * 100
	df['DTERR'] = df['DTERR'] * 100
	df['AUIN'] = df['AUIN'] * 100
	df['AUOUT'] = df['AUOUT'] * 100

	string_table = ''

	# print the resutls in a nice table
	# Determine the maximum width for the ood_dataset column
	max_width = (
		max(len('ood_datasets'), max(len(str(val)) for val in df['ood_dataset'])) + 2
	)

	# Define consistent column widths for the numerical columns
	col_width = 10

	# Construct the header
	string_table += (
		f'{"ood_datasets":<{max_width}}'
		f'{"FPR":<{col_width}}'
		f'{"AUROC":<{col_width}}'
		f'{"DTERR":<{col_width}}'
		f'{"AUIN":<{col_width}}'
		f'{"AUOUT":<{col_width}}'
		'\n'
	)

	# Add a separator line
	string_table += '-' * (max_width + col_width * 5) + '\n'

	# Print rows with consistent widths
	for i in range(len(df)):
		string_table += (
			f'{df.iloc[i]["ood_dataset"]:<{max_width}}'
			f'{df.iloc[i]["FPR"]:<{col_width}.2f}'
			f'{df.iloc[i]["AUROC"]:<{col_width}.2f}'
			f'{df.iloc[i]["DTERR"]:<{col_width}.2f}'
			f'{df.iloc[i]["AUIN"]:<{col_width}.2f}'
			f'{df.iloc[i]["AUOUT"]:<{col_width}.2f}'
			'\n'
		)

	return string_table


ash_methods = ['ashs']  # ["", "ashs", "ashb", "ashp", "scale"]

apes_methodds = ['apesp']  # ["apesscale", "apesp", "apess"]

for ash_methode in tqdm(ash_methods):
	for apes_methode in tqdm(apes_methodds):
		for percentile in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
			if ash_methode == '':
				method_name = f'{apes_methode}@{percentile}'
			else:
				method_name = f'{ash_methode}_{apes_methode}@{percentile}'
			run_single_experiment(
				backbone='resnet50',
				layer='avgpool',
				dataset='imagenet',
				method_name=method_name,
			)
			run_single_experiment(
				backbone='resnet50',
				layer='avgpool',
				dataset='openimage-o',
				method_name=method_name,
			)

			metrics = compute_traditional_ood(
				id_dataset='imagenet',
				ood_datasets=['openimage-o'],
				method_name=method_name,
				print_results=False,
			)
			result_df = DataFrame(metrics)

			table = build_table(result_df)

			result_str = f'Results for {method_name}:\n \n'
			result_str += table
			result_str += '\n\n'

			with open(f'results/ablation_studies/{method_name}.txt', 'a') as f:
				f.write(result_str)
