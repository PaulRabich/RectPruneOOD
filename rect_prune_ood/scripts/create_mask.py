import argparse


def validate_values(values):
	for v in values:
		if v < 1 or v > 19:
			raise argparse.ArgumentTypeError(f'Value {v} is out of range (1–19)')

	duplicates = {v for v in values if values.count(v) > 1}
	if duplicates:
		raise argparse.ArgumentTypeError(
			f'Value list contains duplicates: {sorted(duplicates)}'
		)

	return values

def validate_backbone(backbone):
	if backbone not in ["resnet50", "mobilenet"]:
		raise argparse.ArgumentTypeError(
            f'Backbone {backbone} is not supported. Choose from ["resnet50", "mobilenet"].'
        )
	return backbone


def parse_args():
	parser = argparse.ArgumentParser(
		description='A CLI that accepts a name and a list of integers (1–19).'
	)

	parser.add_argument('--backbone', type=str, required=True, help='Backbone model name.')
	parser.add_argument('--percentile', type=float, required=True, help='Percentile for mask creation.')
	parser.add_argument('--name', type=str, required=True, help='Your name (string).')

	parser.add_argument(
		'--values',
		type=int,
		nargs='+',
		required=True,
		help='List of integers between 1 and 19.',
	)

	args = parser.parse_args()
	args.values = validate_values(args.values)
	args.backbone = validate_backbone(args.backbone)

	return args


def main():
	args = parse_args()
	name = args.name
	values = args.values
	backbone = args.backbone

	# crate mask and save


if __name__ == '__main__':
	main()
