#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
using std::ofstream;
using namespace std;

vector < vector<double> > train_data;
vector <int> train_labels;
vector < vector<double> > test_data;
vector <int> test_labels;
vector < vector<int> > target;
const int number_of_training_data = 60000, number_of_test_data = 10000;
const int labels_dimension = 10;
int neurons_hidden_layer = 30;
int number_of_rows = 28, number_of_columns = 28;
int data_considered = 60000;
int epochs = 10;
double e = 0.01;

//for convolution
int feature_map = 6;
int feature_map_size = 24;
int downsampled_size = 12;
int window_size = 5;



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
				target[i][train_labels[i]] = 1;
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

		for (int i = 0; i < number_of_images; i++)
		{
			for (int r = 0; r < number_of_rows; r++)
			{
				for (int c = 0; c < number_of_columns; c++)
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


float gaussian_random() {
	float g_random = 0;
	for (int i = 0; i < 12; i++)
		g_random += float(rand()) / float(RAND_MAX);
	g_random = (g_random - 6) / 12;
	return g_random;
}
double activate(double r) {
	if (r < 0)
		return 0.01*r;
	else
		return r;
}

double activation_derivative(double x) {
	if (x < 0)
		return 0.01;
	else
		return 1.0;
}

int main()
{
	train_data.resize(number_of_training_data, vector<double>(784));
	train_labels.resize(number_of_training_data);
	test_data.resize(number_of_test_data, vector<double>(784));
	test_labels.resize(number_of_test_data);
	target.resize(number_of_training_data, vector<int>(labels_dimension));

	read_MNIST_data(1);
	read_MNIST_labels(1);
	read_MNIST_data(2);
	read_MNIST_labels(2);

	ofstream outfile("file.txt");

	//For convolution
	vector<double>A, A0 = vector<double>(feature_map);
	vector<vector<vector<double> > > F, F0 = vector<vector<vector<double> >  >(feature_map, vector<vector<double> >(window_size, vector<double>(window_size)));
	vector<vector<vector<double> > > delta_feature_map = vector<vector<vector<double> >  >(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size)));
	vector<vector<vector<double> > > s0, y0 = vector<vector<vector<double> >  >(feature_map, vector<vector<double> >(feature_map_size, vector<double>(feature_map_size)));
	delta_feature_map.resize(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size)));
	A.resize(feature_map);
	A0.resize(feature_map);
	F.resize(feature_map, vector<vector<double> >(window_size, vector<double>(window_size)));
	F0.resize(feature_map, vector<vector<double> >(window_size, vector<double>(window_size)));
	s0.resize(feature_map, vector<vector<double> >(feature_map_size, vector<double>(feature_map_size)));
	y0.resize(feature_map, vector<vector<double> >(feature_map_size, vector<double>(feature_map_size)));


	//weights from hidden to output
	vector<vector<vector<vector<double> > > > W, W0 = vector<vector<vector<vector<double> > > >(neurons_hidden_layer, vector<vector<vector<double> > >(feature_map, vector < vector<double> >(downsampled_size, vector<double>(downsampled_size))));
	W.resize(neurons_hidden_layer, vector < vector<vector<double> > >(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size))));
	W0.resize(neurons_hidden_layer, vector<vector<vector<double> > >(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size))));


	// For pooling
	int max_index_r, max_index_c;
	vector<vector<vector<vector<double> > > > max_index = vector<vector<vector<vector<double> > > >(feature_map, vector<vector<vector<double> > >(downsampled_size, vector < vector<double> >(downsampled_size, vector<double>(2))));
	max_index.resize(feature_map, vector<vector<vector<double> > >(downsampled_size, vector<vector<double> >(downsampled_size, vector<double>(2))));
	vector<vector<vector<double> > > s1 = vector<vector<vector<double> >  >(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size)));
	s1.resize(feature_map, vector<vector<double> >(downsampled_size, vector<double>(downsampled_size)));





	//hidden layer
	vector<double> s2, y2, delta_hidden;
	s2.resize(neurons_hidden_layer);
	y2.resize(neurons_hidden_layer);
	delta_hidden.resize(neurons_hidden_layer);

	//output layer
	vector<double> s3, y3, delta_output;
	s3.resize(10);
	y3.resize(10);
	delta_output.resize(10);



	vector < vector<double> > U, U0, U_trained;
	U.resize(labels_dimension, vector<double>(neurons_hidden_layer));
	U0.resize(labels_dimension, vector<double>(neurons_hidden_layer));
	vector<double> B, B0, B_trained;
	B.resize(neurons_hidden_layer);
	B0.resize(neurons_hidden_layer);
	B_trained.resize(neurons_hidden_layer);
	vector<double> C, C0, C_trained;
	C.resize(10);
	C0.resize(10);
	C_trained.resize(10);



	//For convolution

	// F0, weights for 6 feature maps from 28x28 image
	for (int l = 0; l < feature_map; l++)
		for (int j = 0; j < window_size; j++)
			for (int k = 0; k < window_size; k++)
				F0[l][j][k] = gaussian_random();

	// A0 is the bias for 6 feature map and W0 is the weight from downsampled 12x12 to fully connected NN
	int i, j, k, l, m, n;
	double temp;
	for (i = 0; i < feature_map; i++) {
		for (k = 0; k < downsampled_size; k++)
			for (l = 0; l < downsampled_size; l++)
				for (j = 0; j < neurons_hidden_layer; j++)
					W0[j][i][k][l] = gaussian_random();

		A0[i] = gaussian_random();
	}

	// addded the fully connected NN after the downsampled , B0 is the bias for the neurons in hidden layer
	//initialize B
	for (int i = 0; i < neurons_hidden_layer; ++i) {
		B0[i] = gaussian_random();
	}
	//initialize U
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < neurons_hidden_layer; ++j) {
			U0[i][j] = gaussian_random();
		}
	}
	//initialize C
	for (int i = 0; i < 10; ++i) {
		C0[i] = gaussian_random();
	}

	for (int r = 0; r < epochs; r++) { // for one epoch
		cout << "epochs" << r + 1 << endl;
		//Forward Propagation
		for (int k = 0; k < data_considered; k++) {
			for (l = 0; l < feature_map; l++) {
				// generate feature map
				for (m = 0; m < feature_map_size; m++) {
					for (n = 0; n < feature_map_size; n++) {
						temp = A0[l];

						for (i = 0; i < window_size; i++)
							for (j = 0; j < window_size; j++)
								temp += F0[l][i][j] * train_data[k][(i + m) * 28 + (j + n)];

						s0[l][m][n] = temp;
						y0[l][m][n] = activate(temp);
					}
				}
				// downsize feature map

				//feature map arrays y0 = 6x24x24 and we are going to save the max value in s1=  6x12x12

				for (m = 0; m < downsampled_size; m++) {  // I practically dont need to loop through 24x24, its all about math
					for (n = 0; n < downsampled_size; n++) {
						temp = y0[l][2 * m][2 * n];  // temp = y[0][0][0]
						max_index[l][m][n][0] = 2 * m;
						max_index[l][m][n][1] = 2 * n;

						for (i = m * 2; i < m * 2 + 2; i++)												\
							for (j = n * 2; j < n * 2 + 2; j++)
								if (y0[l][i][j] > temp) {
									temp = y0[l][i][j];
									max_index[l][m][n][0] = i; // saving the row of max value
									max_index[l][m][n][1] = j; // saving the column of max value
								}

						s1[l][m][n] = temp;
					}


				}
			}
			// hidden layer
			for (l = 0; l < neurons_hidden_layer; l++) {
				temp = B0[l];

				for (i = 0; i < feature_map; i++)
					for (m = 0; m < downsampled_size; m++)
						for (n = 0; n < downsampled_size; n++)
							temp += W0[l][i][m][n] * s1[i][m][n];

				s2[l] = temp;
				y2[l] = activate(temp);
			}

			// output layer
			for (l = 0; l < 10; l++) {
				temp = C0[l];

				for (i = 0; i < neurons_hidden_layer; i++)
					temp += U0[l][i] * y2[i];

				s3[l] = temp;
				y3[l] = activate(temp);
			}




			// BACK PROPAGATION
			// output layer
			for (l = 0; l < labels_dimension; l++) {
				// error signals
				delta_output[l] = (y3[l] - target[k][l])*activation_derivative(s3[l]); // for delta dont we need to save the derivative also

				// update U and C
				for (n = 0; n < neurons_hidden_layer; n++) {
					U[l][n] = U0[l][n] - e * delta_output[l] * y2[n];
				}

				C[l] = C0[l] - e * delta_output[l] * 1*activation_derivative(s3[l]); // for bias multiplying 1
			}

			// hidden layer
			for (l = 0; l < neurons_hidden_layer; l++) {
				// error signals
				temp = 0;

				for (i = 0; i < labels_dimension; i++)
					temp += delta_output[i] * U0[i][l];

				delta_hidden[l] = temp * activation_derivative(s2[l]);

				// update W and B
				for (i = 0; i < feature_map; i++)
					for (m = 0; m < downsampled_size; m++)
						for (n = 0; n < downsampled_size; n++)
							W[l][i][m][n] = W0[l][i][m][n] - e * delta_hidden[l] * s1[i][m][n];

				B[l] = B0[l] - e * delta_hidden[l] * 1;
			}

			//update F and A for feature maps
			for (l = 0; l < feature_map; l++) {
				for (m = 0; m < window_size; m++)
					for (n = 0; n < window_size; n++)
						F[l][m][n] = 0;

				A[l] = 0;

				for (m = 0; m < downsampled_size; m++) {
					for (n = 0; n < downsampled_size; n++) {
						temp = 0;

						for (i = 0; i < neurons_hidden_layer; i++)
							temp += delta_hidden[i] * W0[i][l][m][n];

						max_index_r = max_index[l][m][n][0];
						max_index_c = max_index[l][m][n][1];

						delta_feature_map[l][m][n] = temp * activation_derivative(s0[l][max_index_r][max_index_c]);

						for (i = 0; i < window_size; i++)
							for (j = 0; j < window_size; j++)
								F[l][i][j] += delta_feature_map[l][m][n] * train_data[k][(max_index_r + i) * 28 + (max_index_c + j)];// should be 28

						A[l] += delta_feature_map[l][m][n] * 1;
					}
				}

				for (m = 0; m < window_size; m++)
					for (n = 0; n < window_size; n++)
						F[l][m][n] = F0[l][m][n] - e * F[l][m][n];

				A[l] = A0[l] - e * A[l];
			}
			//Update weight
			F0 = F;
			W0 = W;
			A0 = A;

			U0 = U;
			B0 = B;
			C0 = C;
			//cout << "updated weights for epoch " << r + 1 << endl;
		}
		cout << "accuracy testing on train data" << endl;
		int accuracy = 0;
		for (int k = 0; k < number_of_training_data; k++) {
			for (l = 0; l < feature_map; l++) {
				// generate feature map
				for (m = 0; m < feature_map_size; m++) {
					for (n = 0; n < feature_map_size; n++) {
						temp = A0[l];

						for (i = 0; i < window_size; i++)
							for (j = 0; j < window_size; j++)
								temp += F0[l][i][j] * train_data[k][(i + m) * 28 + (j + n)];

						s0[l][m][n] = temp;
						y0[l][m][n] = activate(temp);
					}
				}
				// downsize feature map

				//feature map arrays y0 = 6x24x24 and we are going to save the max value in s1=  6x12x12

				for (m = 0; m < downsampled_size; m++) {  // I practically dont need to loop through 24x24, its all about math
					for (n = 0; n < downsampled_size; n++) {
						temp = y0[l][2 * m][2 * n];
						max_index[l][m][n][0] = 2 * m;
						max_index[l][m][n][1] = 2 * n;

						for (i = m * 2; i < m * 2 + 2; i++)
							for (j = n * 2; j < n * 2 + 2; j++)
								if (y0[l][i][j] > temp) {
									temp = y0[l][i][j];
									max_index[l][m][n][0] = i; // saving the row of max value
									max_index[l][m][n][1] = j; // saving the column of max value
								}

						s1[l][m][n] = temp;
					}


				}
			}
			// hidden layer
			for (l = 0; l < neurons_hidden_layer; l++) {
				temp = B0[l];

				for (i = 0; i < feature_map; i++)
					for (m = 0; m < downsampled_size; m++)
						for (n = 0; n < downsampled_size; n++)
							temp += W0[l][i][m][n] * s1[i][m][n];

				s2[l] = temp;
				y2[l] = activate(temp);
			}

			// output layer
			for (l = 0; l < 10; l++) {
				temp = C0[l];

				for (i = 0; i < neurons_hidden_layer; i++)
					temp += U0[l][i] * y2[i];

				s3[l] = temp;
				y3[l] = activate(temp);
			}
			//count how many are corrects
			// accuracy
			double maxm = *max_element(y3.begin(), y3.end());
			int index = 0;
			for (int k = 0; k < 10; k++) {
				if (y3[k] == maxm) index = k;
			}

			accuracy += train_labels[k] == index ? 1 : 0;



		}
		cout << "number of correct prediction in train data: " << accuracy << endl;;
		cout << "number of incorrect prediction in train data: " << number_of_training_data - accuracy << endl;
		accuracy = ((accuracy*1.0) / number_of_training_data) * 100;
		cout << "accuracy on train data=" << accuracy << "%" << endl;




		cout << "accuracy testing on test data" << endl;
		int accuracy_test = 0;
		for (int k = 0; k < number_of_test_data; k++) {
			for (l = 0; l < feature_map; l++) {
				// generate feature map
				for (m = 0; m < feature_map_size; m++) {
					for (n = 0; n < feature_map_size; n++) {
						temp = A0[l];

						for (i = 0; i < window_size; i++)
							for (j = 0; j < window_size; j++)
								temp += F0[l][i][j] * test_data[k][(i + m) * 28 + (j + n)];

						s0[l][m][n] = temp;
						y0[l][m][n] = activate(temp);
					}
				}
				// downsize feature map

				//feature map arrays y0 = 6x24x24 and we are going to save the max value in s1=  6x12x12

				for (m = 0; m < downsampled_size; m++) {  // I practically dont need to loop through 24x24, its all about math
					for (n = 0; n < downsampled_size; n++) {
						temp = y0[l][2 * m][2 * n];
						max_index[l][m][n][0] = 2 * m;
						max_index[l][m][n][1] = 2 * n;

						for (i = m * 2; i < m * 2 + 2; i++)
							for (j = n * 2; j < n * 2 + 2; j++)
								if (y0[l][i][j] > temp) {
									temp = y0[l][i][j];
									max_index[l][m][n][0] = i; // saving the row of max value
									max_index[l][m][n][1] = j; // saving the column of max value
								}

						s1[l][m][n] = temp;
					}


				}
			}
			// hidden layer
			for (l = 0; l < neurons_hidden_layer; l++) {
				temp = B0[l];

				for (i = 0; i < feature_map; i++)
					for (m = 0; m < downsampled_size; m++)
						for (n = 0; n < downsampled_size; n++)
							temp += W0[l][i][m][n] * s1[i][m][n];

				s2[l] = temp;
				y2[l] = activate(temp);
			}

			// output layer
			for (l = 0; l < 10; l++) {
				temp = C0[l];

				for (i = 0; i < neurons_hidden_layer; i++)
					temp += U0[l][i] * y2[i];

				s3[l] = temp;
				y3[l] = activate(temp);
			}
			//count how many are corrects
			// accuracy
			double maxm = *max_element(y3.begin(), y3.end());
			int index = 0;
			for (int k = 0; k < 10; k++) {
				if (y3[k] == maxm) index = k;
			}

			accuracy_test += test_labels[k] == index ? 1 : 0;



		}
		cout << "number of correct prediction in test data: " << accuracy_test << endl;;
		cout << "number of incorrect prediction in test data: " << number_of_test_data - accuracy_test << endl;
		accuracy_test = ((accuracy_test*1.0) / number_of_test_data) * 100;
		cout << "accuracy on test data=" << accuracy_test << "%" << endl;
	}

	getchar();
	return 0;

}





