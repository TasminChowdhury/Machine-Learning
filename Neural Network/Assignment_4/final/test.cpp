#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <string.h>

using namespace std;

vector < vector<double> > train_data;
vector <int> train_labels;
vector < vector<double> > test_data;
vector <int> test_labels;
vector < vector<double> > w0_trained;
vector < vector<double> > w1_trained;
vector < double > b0_trained;
vector < double > b1_trained;
vector < double > s0_test;
vector < double > y0_test;
vector < double > s1_test;
vector < double > y1_test;

const int number_of_training_data = 60000, number_of_test_data = 10000;
const int labels_dimension = 10, data_dimension = 784;
int neurons_hidden_layer = 50;
int number_of_rows = 28, number_of_columns = 28;
#include "trained_weights.h"
int reverse_int(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_MNIST_labels(int k) {
	ifstream file;
	if (k == 1) {
		file.open("train-labels-idx1-ubyte", ios::binary);
	}
	else {
		file.open("t10k-labels-idx1-ubyte", ios::binary);
	}

	if (file.is_open()) {
		int magic_number = 0, number_of_items = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		file.read((char*)&number_of_items, sizeof(number_of_items));
		number_of_items = reverse_int(number_of_items);

		for (int i = 0; i < number_of_items; i++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (k == 1) {
				train_labels[i] = (int)temp;
			}
			else {
				test_labels[i] = (int)temp;
			}
		}
	}
	else cout << "cant";
}

void read_MNIST_data(int k) {
	ifstream file;
	if (k == 1) {
		file.open("train-images-idx3-ubyte", ios::binary);
	}
	else {
		file.open("t10k-images-idx3-ubyte", ios::binary);
	}
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverse_int(number_of_images);
		file.read((char*)&number_of_rows, sizeof(number_of_rows));
		number_of_rows = reverse_int(number_of_rows);
		file.read((char*)&number_of_columns, sizeof(number_of_columns));
		number_of_columns = reverse_int(number_of_columns);

		for (int i = 0; i<number_of_images; i++)
		{
			for (int r = 0; r<number_of_rows; r++)
			{
				for (int c = 0; c<number_of_columns; c++)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					if (k == 1) {
						train_data[i][(number_of_rows*r) + c] = (float)(temp*1.0) / 255;
					}
					else {
						test_data[i][(number_of_rows*r) + c] = (float)(temp*1.0) / 255;
					}
				}
			}
		}
	}
	else cout << "cant";
}


void test() {
	cout << "\ntesting on training data\n";
	float temp;
	float accuracy = 0;
	for (int i = 0; i < number_of_training_data; i++)
	{
		// FORWARD PROPAGATION

		for (int j = 0; j<neurons_hidden_layer; j++) {
			for (int k = 0; k<data_dimension; k++) {
				s0_test[j] += w0_trained[k][j] * train_data[i][k];
			}
		}

		for (int j = 0; j<neurons_hidden_layer; j++)
			s0_test[j] += b0_trained[j];

		for (int j = 0; j<neurons_hidden_layer; j++)
			y0_test[j] = tanh(s0_test[j]);



		for (int j = 0; j<labels_dimension; j++) {
			for (int k = 0; k<neurons_hidden_layer; k++) {
				s1_test[j] += w1_trained[k][j] * y0_test[k];
			}
		}

		for (int j = 0; j<labels_dimension; j++)
			s1_test[j] += b1_trained[j];

		for (int j = 0; j<labels_dimension; j++)
			y1_test[j] = tanh(s1_test[j]);

		double maxm = *max_element(y1_test.begin(), y1_test.end());
		int index = 0;
		for (int k = 0; k<10; k++) {
			if (y1_test[k] == maxm) index = k;
		}

		accuracy += train_labels[i] == index ? 1 : 0;

		fill(s0_test.begin(), s0_test.end(), 0);
		fill(y0_test.begin(), y0_test.end(), 0);
		fill(s1_test.begin(), s1_test.end(), 0);
		fill(y1_test.begin(), y1_test.end(), 0);
	}
	accuracy = ((accuracy*1.0) / number_of_training_data) * 100;
	cout << "accuracy=" << accuracy << "%" << endl;


	cout << "\ntesting on test data\n";
	accuracy = 0;
	for (int i = 0; i < number_of_test_data; i++)
	{
		for (int j = 0; j<neurons_hidden_layer; j++) {
			for (int k = 0; k<data_dimension; k++) {
				s0_test[j] += w0_trained[k][j] * test_data[i][k];
			}
		}

		for (int j = 0; j<neurons_hidden_layer; j++)
			s0_test[j] += b0_trained[j];

		for (int j = 0; j<neurons_hidden_layer; j++)
			y0_test[j] = tanh(s0_test[j]);



		for (int j = 0; j<labels_dimension; j++) {
			for (int k = 0; k<neurons_hidden_layer; k++) {
				s1_test[j] += w1_trained[k][j] * y0_test[k];
			}
		}

		for (int j = 0; j<labels_dimension; j++)
			s1_test[j] += b1_trained[j];

		for (int j = 0; j<labels_dimension; j++)
			y1_test[j] = tanh(s1_test[j]);

		double maxm = *max_element(y1_test.begin(), y1_test.end());
		int index = 0;
		for (int k = 0; k<10; k++) {
			if (y1_test[k] == maxm) index = k;
		}
		accuracy += test_labels[i] == index ? 1 : 0;

		fill(s0_test.begin(), s0_test.end(), 0);
		fill(y0_test.begin(), y0_test.end(), 0);
		fill(s1_test.begin(), s1_test.end(), 0);
		fill(y1_test.begin(), y1_test.end(), 0);

	}
	accuracy = ((accuracy*1.0) / number_of_test_data) * 100;
	cout << "accuracy=" << accuracy << "%" << endl;


}

int main()
{

	s0_test.resize(neurons_hidden_layer);
	y0_test.resize(neurons_hidden_layer);
	s1_test.resize(labels_dimension);
	y1_test.resize(labels_dimension);
	w0_trained.resize(data_dimension, vector<double>(neurons_hidden_layer));
	b0_trained.resize(neurons_hidden_layer);
	w1_trained.resize(neurons_hidden_layer, vector<double>(labels_dimension));
	b1_trained.resize(labels_dimension);

	train_data.resize(number_of_training_data, vector<double>(data_dimension));
	train_labels.resize(number_of_training_data);
	test_data.resize(number_of_test_data, vector<double>(data_dimension));
	test_labels.resize(number_of_test_data);

	set_weights();

	read_MNIST_data(1);
	read_MNIST_labels(1);
	read_MNIST_data(2);
	read_MNIST_labels(2);

	test();
	getchar();
	return 0;
}
