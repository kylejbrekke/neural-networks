from Experiment import Experiment
import sys


class Main:

	@staticmethod
	def main(args):
		file_name = args[0]
		class_header = args[1]
		learning_rate = float(args[2])
		verbose = True if str.lower(args[3]) == 'true' else False
		use_momentum = True if str.lower(args[4]) == 'true' else False
		regression = True if str.lower(args[5]) == 'true' else False
		print(regression)

		experiment = Experiment(('../datasets/%s' % file_name), class_header)
		if len(args) > 6:
			discrete_columns = args[6:]
			experiment.clean(discrete_columns=discrete_columns)
		experiment.experiment(learning_rate, regression=regression, verbose=verbose, momentum=use_momentum)


if __name__ == '__main__':
	Main.main(sys.argv[1:])
