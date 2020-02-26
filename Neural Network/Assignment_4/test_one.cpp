#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

const string training_data_file = "train-images-idx3-ubyte", test_data_file = "t10k-images-idx3-ubyte";
const string training_labels_file = "train-labels-idx1-ubyte", test_labels_file = "t10k-labels-idx1-ubyte";
const int number_of_training_data = 60000, number_of_test_data = 10000;
const int labels_dimension = 10, data_dimension = 784;
int neurons_hidden_layer = 30;
int number_of_rows = 28, number_of_columns = 28;

float **w0_trained, *b0_trained, **w1_trained, *b1_trained;
float *s0_test, *y0_test, *s1_test, *y1_test;

#include "trained_weights.h"

void allocate_data(float **&train_data, int *&train_labels, float **&test_data, int *&test_labels) {
    s0_test = new float[neurons_hidden_layer];
    y0_test = new float[neurons_hidden_layer];
    s1_test = new float[labels_dimension];
    y1_test = new float[labels_dimension];

    train_data = new float*[number_of_training_data];
    train_labels = new int[number_of_training_data];

    for(int i = 0; i < number_of_training_data; ++i)
        train_data[i] = new float[data_dimension];

    test_data = new float*[number_of_test_data];
    test_labels = new int[number_of_test_data];

    for(int i = 0; i < number_of_test_data; ++i)
        test_data[i] = new float[data_dimension];
}

void allocate_weights() {
    w0_trained = new float*[data_dimension];
    b0_trained = new float[neurons_hidden_layer];

    for (int i=0; i< data_dimension; ++i)
        w0_trained[i] = new float[neurons_hidden_layer];

    w1_trained = new float*[neurons_hidden_layer];
    b1_trained = new float[labels_dimension];

    for (int i=0; i< neurons_hidden_layer; ++i)
        w1_trained[i] = new float[labels_dimension];
}

void free_weights() {
	for(int i = 0; i < data_dimension; ++i) {
		delete [] w0_trained[i];
	}
	delete [] w0_trained;
	delete [] b0_trained;

	for (int i=0; i<neurons_hidden_layer; ++i) {
		delete [] w1_trained[i];
	}
	delete [] w1_trained;
	delete [] b1_trained;
}

int reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void read_MNIST_labels(string label_file, int *vec) {
    ifstream file (label_file, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, number_of_items = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_items, sizeof(number_of_items));
        number_of_items = reverse_int(number_of_items);

        for(int i = 0; i < number_of_items; i++)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (int)temp;
        }
    }
}

void read_MNIST_data(string data_file, float **array) {
    ifstream file(data_file, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number= reverse_int(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images= reverse_int(number_of_images);
        file.read((char*)&number_of_rows, sizeof(number_of_rows));
        number_of_rows= reverse_int(number_of_rows);
        file.read((char*)&number_of_columns, sizeof(number_of_columns));
        number_of_columns= reverse_int(number_of_columns);

        for(int i=0; i<number_of_images; i++)
        {
            for(int r=0; r<number_of_rows; r++)
            {
                for(int c=0; c<number_of_columns; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    array[i][(number_of_rows*r)+c] = (float) (temp*1.0)/255;
                }
            }
        }
    }
}

void free_data(float **train_data, int *train_labels, float **test_data, int *test_labels) {
    for(int i = 0; i < number_of_training_data; ++i)
        delete [] train_data[i];

    delete [] train_data;
    delete [] train_labels;

    for (int i=0; i<number_of_test_data; i++)
        delete [] test_data[i];

    delete [] test_data;
    delete [] test_labels;
 }

 void multiply(float **weights, float *datum, float *result, int dim_inputs, int dim_layer) {
    for (int j=0; j<dim_layer; j++) {
        result[j] = 0;
        for (int i=0; i<dim_inputs; i++) {
            result[j] += weights[i][j]*datum[i];
        }
    }
}

// Adds vector 1 and 2 and keeps the result in vector 1
void add(float *vec1, float *vec2, int dim) {
    for (int i=0; i<dim; i++)
        vec1[i] += vec2[i];
}

void activate(float *input, float *output, int dim) {
    for(int i=0; i<dim; i++)
        output[i] = tanh(input[i]);
}

void test(float **train_data, int *train_labels, float **test_data, int *test_labels, int test_in, int test_index) {
    float prediction;
    if (test_in==0) {
        cout << "\ntest on train data\n";
        multiply(w0_trained, train_data[test_index], s0_test, data_dimension, neurons_hidden_layer);
    } else if (test_in==1) {
        cout << "\ntest on test data\n";
        multiply(w0_trained, test_data[test_index], s0_test, data_dimension, neurons_hidden_layer);
    }

    add(s0_test, b0_trained, neurons_hidden_layer);
    activate(s0_test, y0_test, neurons_hidden_layer);

    multiply(w1_trained, y0_test, s1_test, neurons_hidden_layer, labels_dimension);
    add(s1_test, b1_trained, labels_dimension);
    activate(s1_test, y1_test, labels_dimension);

    prediction = max_element(y1_test, y1_test+labels_dimension) - y1_test;

    for(int r=0; r<number_of_rows; r++)
    {
        for(int c=0; c<number_of_columns; c++)
        {
            if (test_in==0)
                cout << train_data[test_index][(number_of_rows*r)+c]*255 <<"\t";
            else if (test_in==1)
                cout << test_data[test_index][(number_of_rows*r)+c]*255 <<"\t";
        }
        cout << endl << endl;
    }

    cout << "The digit is predicted as:\t" << prediction << endl;

    delete [] s0_test;
    delete [] y0_test;
    delete [] s1_test;
    delete [] y1_test;
}

int main()
{
    float **train_data, **test_data;
    int *train_labels, *test_labels;

    allocate_data(train_data, train_labels, test_data, test_labels);
    allocate_weights();
    set_weights();

    read_MNIST_data(training_data_file, train_data);
    read_MNIST_labels(training_labels_file, train_labels);
    read_MNIST_data(test_data_file, test_data);
    read_MNIST_labels(test_labels_file, test_labels);

    int test_in = 1; // 1 for test and 0 for train
    int test_index = 100; // for train < 60000, for test < 10000
    test(train_data, train_labels, test_data, test_labels, test_in, test_index);

    free_data(train_data, train_labels, test_data, test_labels);
    free_weights();
    return 0;
}
