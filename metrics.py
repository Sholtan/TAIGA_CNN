
import numpy as np
import matplotlib.pyplot as plt

class Metrics:

	def draw_precision_from_suppression(self, y_probabilities, test_labels, gate_start_value, gate_n):
		y_probabilities = y_probabilities.reshape((-1,))
		test_labels = test_labels.reshape((-1,))

		m = test_labels.shape[0]
	
		gate_values_array = np.linspace(gate_start_value, 1., gate_n)

		precision_array = []
		suppression_array = []

		i = 0
		for gate in gate_values_array:
			#print("gate:", gate)
			y_after_gate = y_probabilities.copy()
			y_after_gate[y_after_gate >= gate] = 1
			y_after_gate[y_after_gate < gate] = 0

			true_positives = 0
			false_negatives = 0

			false_positives = 0
			true_negatives = 0

			for i in range(len(test_labels)):
				if (y_after_gate[i] == 1 and test_labels[i] == 1):
					true_positives += 1
				elif (y_after_gate[i] == 0 and test_labels[i] == 1):
					false_negatives += 1
				elif (y_after_gate[i] == 1 and test_labels[i] == 0):
					false_positives += 1
				elif (y_after_gate[i] == 0 and test_labels[i] == 0):
					true_negatives += 1
				else:
					raise Exception("Something wrong with y_after_gate")
			if i == 0:
				print("true_positives:", true_positives)
				print("false_negatives:", false_negatives)
				print("false_positives:", false_positives)
				print("true_negatives:", true_negatives)
			precision = true_positives / (true_positives + false_negatives)

			# Inverse of False Omission Rate:
			suppression = (false_positives + true_negatives) / false_positives


			precision_array.append(precision)
			suppression_array.append(suppression)
			i += 1


		plt.plot(precision_array, suppression_array, linewidth=3)
		plt.ylabel('Suppression', fontsize=18)
		plt.grid(True, which='both')
		plt.xlabel('Precision', fontsize=18)
		plt.title("ROC", fontsize=20)

		plt.show()

		return precision_array, suppression_array