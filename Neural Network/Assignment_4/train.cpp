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
int training_data_considered = 100;
int epochs = 30;
// Superparameter eta
float e = 0.01;

int number_of_rows = 28, number_of_columns = 28;

float **w0_trained, *b0_trained, **w1_trained, *b1_trained, *s0_test, *y0_test, *s1_test, *y1_test;

void allocate_data(float **&train_data, int *&train_labels, int **&train_labels_vectorized, float **&test_data, int *&test_labels, int **&test_labels_vectorized) {
    train_data = new float*[number_of_training_data];
    train_labels = new int[number_of_training_data];
    train_labels_vectorized = new int*[number_of_training_data];

    for(int i = 0; i < number_of_training_data; ++i) {
        train_data[i] = new float[data_dimension];
        train_labels_vectorized[i] = new int[labels_dimension];
    }

    test_data = new float*[number_of_test_data];
    test_labels = new int[number_of_test_data];
    test_labels_vectorized = new int*[number_of_test_data];

    for(int i = 0; i < number_of_test_data; ++i) {
        test_data[i] = new float[data_dimension];
        test_labels_vectorized[i] = new int[labels_dimension];
    }
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
            cout<<vec[i];
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
                    //cout<<array[i][(number_of_rows*r)+c];
                }
            }
        }
    }
}

void vectorize_labels(int **vectorized_labels, int *label, int type=0) {
    int n;
    n = type==0 ? number_of_training_data : number_of_test_data;
    for (int i=0; i<n; i++) {
        for (int j=0; j<labels_dimension; j++) {
            vectorized_labels[i][j] = j==label[i] ? 1 : 0;
        }
    }
}

void read_data(float **train_data, int *train_labels, int **train_labels_vectorized, float **test_data, int *test_labels, int **test_labels_vectorized) {
    read_MNIST_data(training_data_file, train_data);
    read_MNIST_labels(training_labels_file, train_labels);
    vectorize_labels(train_labels_vectorized, train_labels, 0);

    read_MNIST_data(test_data_file, test_data);
    read_MNIST_labels(test_labels_file, test_labels);
    vectorize_labels(test_labels_vectorized, test_labels, 1);
}

void free_data(float **train_data, int *train_labels, int **train_labels_vectorized, float **test_data, int *test_labels, int **test_labels_vectorized) {
    for(int i = 0; i < number_of_training_data; ++i) {
        delete [] train_data[i];
        delete [] train_labels_vectorized[i];
    }
    delete [] train_data;
    delete [] train_labels;
    delete [] train_labels_vectorized;

    for (int i=0; i<number_of_test_data; i++) {
        delete [] test_data[i];
        delete [] test_labels_vectorized[i];
    }
    delete [] test_data;
    delete [] test_labels;
    delete [] test_labels_vectorized;
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

void allocate_weights(float **&w0, float **&w00, float *&b0, float *&b00, float *&s0, float *&y0, float **&w1, float **&w10, float *&b1, float *&b10, float *&s1, float *&y1) {
    w0 = new float*[data_dimension];
    w00 = new float*[data_dimension];
    b0 = new float[neurons_hidden_layer];
    b00 = new float[neurons_hidden_layer];
    s0 = new float[neurons_hidden_layer];
    y0 = new float[neurons_hidden_layer];


    for (int i=0; i< data_dimension; ++i) {
        w0[i] = new float[neurons_hidden_layer];
        w00[i] = new float[neurons_hidden_layer];
    }

    w1 = new float*[neurons_hidden_layer];
    w10 = new float*[neurons_hidden_layer];
    b1 = new float[labels_dimension];
    b10 = new float[labels_dimension];
    s1 = new float[labels_dimension];
    y1 = new float[labels_dimension];


    for (int i=0; i< neurons_hidden_layer; ++i) {
        w1[i] = new float[labels_dimension];
        w10[i] = new float[labels_dimension];
    }
}

float gaussian_random() {
    float g_random = 0;
    for (int i=0; i<12; i++)
        g_random += float(rand()) / float(RAND_MAX);
    g_random = (g_random-6)/12;
    return g_random;
}

void init_weights_random(float **weights, int dim_inputs, int dim_layer) {
    for (int i=0; i<dim_inputs; i++) {
        for (int j=0; j<dim_layer; j++) {
            //weights[i][j] = (float(rand()) / float(RAND_MAX))*2*0.12 - 0.12;
            weights[i][j] = gaussian_random();
        }
    }
}

void init_weights_random(float *weights, int dim) {
    for (int i=0; i<dim; i++)
        //weights[i] = (float(rand()) / float(RAND_MAX))*2*0.12 - 0.12;
        weights[i] = gaussian_random();
}

void init_weights(float **w00, float *b00, float **w10, float *b10) {
    init_weights_random(w00, data_dimension, neurons_hidden_layer);
    init_weights_random(b00, neurons_hidden_layer);
    init_weights_random(w10, neurons_hidden_layer, labels_dimension);
    init_weights_random(b10, labels_dimension);
}

void update_weights(float **old, float **new_, int dim1, int dim2) {
    for (int i=0; i<dim1; i++) {
        for (int j=0; j<dim2; j++) {
            old[i][j] = new_[i][j];
        }
    }
}

void update_bias(float *old, float *new_, int dim) {
    for (int i=0; i<dim; i++)
        old[i] = new_[i];
}

void free_weights(float **w0, float *b0, float **w1, float *b1) {
    for(int i = 0; i < data_dimension; ++i) {
        delete [] w0[i];
    }
    delete [] w0;
    delete [] b0;

    for (int i=0; i<neurons_hidden_layer; ++i) {
        delete [] w1[i];
    }
    delete [] w1;
    delete [] b1;
}

void free_remaining_weights(float **w00, float *b00, float **w10, float *b10, float *s0, float *y0, float *s1, float *y1) {
    for(int i = 0; i < data_dimension; ++i) {
        delete [] w00[i];
    }
    delete [] w00;
    delete [] b00;
    delete [] s0;
    delete [] y0;

    for (int i=0; i<neurons_hidden_layer; ++i) {
        delete [] w10[i];
    }
    delete [] w10;
    delete [] b10;
    delete [] s1;
    delete [] y1;
}

void activate(float *input, float *output, int dim) {
    for(int i=0; i<dim; i++)
        output[i] = tanh(input[i]);
}

void save_trained_weights(float **w00, float *b00, float **w10, float *b10) {
    ofstream trained_weights("trained_weights.h");

    trained_weights << "#ifndef TRAINED_WEIGHTS_H\n#define TRAINED_WEIGHTS_H\n\n";
    trained_weights << "void set_weights() {\n";

    for (int i=0; i<data_dimension; i++) {
        for (int j=0; j<neurons_hidden_layer; j++) {
            trained_weights << "\tw0_trained[" << i << "][" << j << "] = " << w00[i][j] << ";\n";
        }
    }

    for (int i=0; i<neurons_hidden_layer; i++) {
        trained_weights << "\tb0_trained[" << i << "] = " << b00[i] << ";\n";
    }

    for (int i=0; i<neurons_hidden_layer; i++) {
        for (int j=0; j<labels_dimension; j++) {
            trained_weights << "\tw1_trained[" << i << "][" << j << "] = " << w10[i][j] << ";\n";
        }
    }

    for (int i=0; i<labels_dimension; i++) {
        trained_weights << "\tb1_trained[" << i << "] = " << b10[i] << ";\n";
    }

    trained_weights << "}\n\n#endif\n";
    trained_weights.close();
}

void train(float **data, int *labels_, int **labels) {
    cout << "---- train start\n";

    // Declaration and initialization of parameters
    ofstream outfile("train_tanh_online.txt");
    ofstream plot("plot.csv");
    outfile << "prediction:truth:error " << endl;

    float **w0, **w00, *b0, *b00, *s0, *y0, **w1, **w10, *b1, *b10, *s1, *y1;

    allocate_weights(w0, w00, b0, b00, s0, y0, w1, w10, b1, b10, s1, y1);
    init_weights(w00, b00, w10, b10);

    // Initial training of the neural network
    cout << "-- init train start\n";
    for (int i = 0; i < training_data_considered; i++)
	{
        // FORWARD PROPAGATION
        multiply(w0, data[i], s0, data_dimension, neurons_hidden_layer);
        add(s0, b0, neurons_hidden_layer);
        activate(s0, y0, neurons_hidden_layer);

        multiply(w1, y0, s1, neurons_hidden_layer, labels_dimension);
        add(s1, b1, labels_dimension);
        activate(s1, y1, labels_dimension);


        // BACK PROPAGATION
        // b1
        for (int j=0; j<labels_dimension; j++) {
            b1[j] = b10[j] - e*(y1[j] - labels[i][j])*(1 - y1[j]*y1[j]);
        }

        // w1
        for (int j=0; j<neurons_hidden_layer; j++) {
            for (int k=0; k<labels_dimension; k++) {
                w1[j][k] = w10[j][k] - e*(y1[k] - labels[i][k])*(1 - y1[k]*y1[k])*y0[j];
            }
        }

        // b0
        float delta;
        for (int j=0; j<neurons_hidden_layer; j++) {
            delta = 0;
            for (int k=0; k<labels_dimension; k++) {
                delta += (y1[k] - labels[i][k])*(1 - y1[k]*y1[k])*(1 - y0[j]*y0[j])*w10[j][k];
            }
            b1[j] = b10[j] - e*(delta);
        }

        // w0
        for (int j=0; j<data_dimension; j++) {
            for (int k=0; k<neurons_hidden_layer; k++) {
                delta = 0;
                for (int l=0; l<labels_dimension; l++) {
                    delta += (y1[l] - labels[i][l])*(1 - y1[l]*y1[l])*(1 - y0[k]*y0[k])*w10[k][l]*data[i][j];
                }
                w0[j][k] = w00[j][k] - e*(delta);
            }
        }

        for (int j=0; j<labels_dimension; j++)
            outfile << y1[j] << ":" << labels[i][j] << ":" << y1[j] - labels[i][j] << ", ";
        outfile << endl;

        // Train further
        update_weights(w00, w0, data_dimension, neurons_hidden_layer);
        update_bias(b00, b0, neurons_hidden_layer);
        update_weights(w10, w1, neurons_hidden_layer, labels_dimension);
        update_bias(b10, b1, labels_dimension);
	}

	cout << "-- init train end\n\n";

	// Epochs needed to finish the training
	cout << "-- epochs train start\n";
	for (int r = 0; r < epochs; r++)
	{
        outfile << "BEGINING OF EPOCH " << r + 1 << endl;
		for (int i = 0; i < training_data_considered; i++)
		{
            // FORWARD PROPAGATION
            multiply(w0, data[i], s0, data_dimension, neurons_hidden_layer);
            add(s0, b0, neurons_hidden_layer);
            activate(s0, y0, neurons_hidden_layer);

            multiply(w1, y0, s1, neurons_hidden_layer, labels_dimension);
            add(s1, b1, labels_dimension);
            activate(s1, y1, labels_dimension);

            // BACK PROPAGATION
            // b1
            for (int j=0; j<labels_dimension; j++) {
                b1[j] = b10[j] - e*(y1[j] - labels[i][j])*(1 - y1[j]*y1[j]);
            }

            // w1
            for (int j=0; j<neurons_hidden_layer; j++) {
                for (int k=0; k<labels_dimension; k++) {
                    w1[j][k] = w10[j][k] - e*(y1[k] - labels[i][k])*(1 - y1[k]*y1[k])*y0[j];
                }
            }

            // b0
            float delta;
            for (int j=0; j<neurons_hidden_layer; j++) {
                delta = 0;
                for (int k=0; k<labels_dimension; k++) {
                    delta += (y1[k] - labels[i][k])*(1 - y1[k]*y1[k])*(1 - y0[j]*y0[j])*w10[j][k];
                }
                b1[j] = b10[j] - e*(delta);
            }

            // w0
            for (int j=0; j<data_dimension; j++) {
                for (int k=0; k<neurons_hidden_layer; k++) {
                    delta = 0;
                    for (int l=0; l<labels_dimension; l++) {
                        delta += (y1[l] - labels[i][l])*(1 - y1[l]*y1[l])*(1 - y0[k]*y0[k])*w10[k][l]*data[i][j];
                    }
                    w0[j][k] = w00[j][k] - e*(delta);
                }
            }

            float SE = 0;
            for (int j=0; j<labels_dimension; j++) {
                outfile << y1[j] << ":" << labels[i][j] << ":" << y1[j] - labels[i][j] << ", ";
                SE += (y1[j] - labels[i][j])*(y1[j] - labels[i][j]);
            }
            outfile << "SSE = " << SE << endl;
            plot << SE << "," << endl;

            // train further
            update_weights(w00, w0, data_dimension, neurons_hidden_layer);
            update_bias(b00, b0, neurons_hidden_layer);
            update_weights(w10, w1, neurons_hidden_layer, labels_dimension);
            update_bias(b10, b1, labels_dimension);
		}
		outfile << " End of Epoch " << r+1  << endl;
		cout << " End of Epoch " << r+1 << endl;
	}
    cout << "epoch end\n";
    w0_trained = w00;
    b0_trained = b00;
    w1_trained = w10;
    b1_trained = b10;
    s0_test = s0;
    y0_test = y0;
    s1_test = s1;
    y1_test = y1;


    // test
    cout << "\ntest on train data\n";
    float temp;
    float accuracy=0;
    for (int i = 0; i < training_data_considered; i++)
    {
        // FORWARD PROPAGATION
        multiply(w0, data[i], s0, data_dimension, neurons_hidden_layer);
        add(s0, b0, neurons_hidden_layer);
        activate(s0, y0, neurons_hidden_layer);

        multiply(w1, y0, s1, neurons_hidden_layer, labels_dimension);
        add(s1, b1, labels_dimension);
        activate(s1, y1, labels_dimension);

        temp = max_element(y1, y1+labels_dimension) - y1;

        accuracy += labels_[i] == temp ? 1 : 0;
        //cout << labels_[i] << " " << temp << endl;
    }
    accuracy = ((accuracy*1.0)/training_data_considered)*100;
    cout << "accuracy=" << accuracy << "%" << endl;

    // free weights
    free_weights(w0, b0, w1, b1);
    outfile.close();
    plot.close();
    save_trained_weights(w00, b00, w10, b10);

    cout << "---- train end\n";
 }

 void test(float **data, int *labels_, int **labels ) {
    cout << "\ntesting....\n";
    float temp;
    float accuracy=0;
    for (int i = 0; i < number_of_test_data; i++)
    {
        // FORWARD PROPAGATION
        multiply(w0_trained, data[i], s0_test, data_dimension, neurons_hidden_layer);
        add(s0_test, b0_trained, neurons_hidden_layer);
        activate(s0_test, y0_test, neurons_hidden_layer);

        multiply(w1_trained, y0_test, s1_test, neurons_hidden_layer, labels_dimension);
        add(s1_test, b1_trained, labels_dimension);
        activate(s1_test, y1_test, labels_dimension);

        temp = max_element(y1_test, y1_test+labels_dimension) - y1_test;

        accuracy += labels_[i] == temp ? 1 : 0;
        //cout << labels_[i] << " " << temp << endl;
    }
    accuracy = ((accuracy*1.0)/number_of_test_data)*100;
    cout << "accuracy=" << accuracy << "%" << endl;
}

int main()
{
    float **train_data, **test_data;
    int *train_labels, **train_labels_vectorized, *test_labels, **test_labels_vectorized;

    allocate_data(train_data, train_labels, train_labels_vectorized, test_data, test_labels, test_labels_vectorized);
    read_data(train_data, train_labels, train_labels_vectorized, test_data, test_labels, test_labels_vectorized);

    train(train_data, train_labels, train_labels_vectorized);
    //test(test_data, test_labels, test_labels_vectorized);

    free_data(train_data, train_labels, train_labels_vectorized, test_data, test_labels, test_labels_vectorized);
    free_remaining_weights(w0_trained, b0_trained, w1_trained, b1_trained, s0_test, y0_test, s1_test, y1_test);
    return 0;
}
