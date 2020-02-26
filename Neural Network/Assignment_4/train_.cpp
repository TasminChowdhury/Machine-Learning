#include <iostream>
#include <vector>
#include <fstream>
using std::ofstream;
#include <cmath>
using namespace std;
const int number_of_training_data = 60000, number_of_test_data = 10000;
const int labels_dimension = 10, data_dimension = 784;
int neurons_hidden_layer = 30;
int number_of_rows = 28, number_of_columns = 28;

vector < vector<double> > train_data;
vector <int> train_labels;
vector < vector<double> > test_data;
vector <int> test_labels;

vector < vector<int> > target;
int data_considered=60000;
int epochs= 30;

void allocate(){
    train_data.resize(number_of_training_data, vector<double>(data_dimension));
    train_labels.resize(number_of_training_data);
    test_data.resize(number_of_test_data, vector<double>(data_dimension));
    test_labels.resize(number_of_test_data);
    target.resize(number_of_training_data, vector<int>(labels_dimension));

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

 void read_MNIST_labels_train() {
    ifstream file;
    file.open("train-labels-idx1-ubyte", ios::binary);
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
            train_labels[i]= (int)temp;
            target[i][train_labels[i]]=1;
            //cout<<train_labels[i]<<" ";
            //for(int j=0; j<10;j++)
              //  cout<<target[i][j]<<" ";
            //cout<<endl;
        }
    }
    else cout<<"cant";
}
 void read_MNIST_labels_test() {
    ifstream file;
    file.open("t10k-labels-idx1-ubyte", ios::binary);

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
            test_labels[i]= (int)temp;
        }
    }
    else cout<<"cant";
}
 void read_MNIST_data_train() {
    ifstream file;
    file.open("train-images-idx3-ubyte", ios::binary);

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
                    train_data[i][(number_of_rows*r)+c] = (float) (temp*1.0)/255;
                    //cout<<array[i][(number_of_rows*r)+c]<<" ";
                }
            }
        }
    }
    else cout<<"cant";
}
void read_MNIST_data_test() {//nao lagte pare
    ifstream file;
    file.open("t10k-images-idx3-ubyte", ios::binary);

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
                    test_data[i][(number_of_rows*r)+c] = (float) (temp*1.0)/255;
                    //cout<<array[i][(number_of_rows*r)+c]<<" ";
                }
            }
        }
    }
    else cout<<"cant";

}

void save_trained_weights( vector < vector<double> > &w00, vector<double> &b00,  vector < vector<double> > &w10, vector<double> &b10) {
    int data_dimension = 784, neurons_hidden_layer = 30, labels_dimension = 10;

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
float gaussian_random() {
    float g_random = 0;
    for (int i=0; i<12; i++)
        g_random += float(rand()) / float(RAND_MAX);
    g_random = (g_random-6)/12;
    return g_random;
}
void train(){
    ofstream outfile("file2.txt");
    ofstream plot("plot.csv");
	vector < vector<double> > W, W0,w0_trained;
	W.resize(784, vector<double>(30));
	W0.resize(784, vector<double>(30));
	w0_trained.resize(784, vector<double>(30));
	vector < vector<double> > U, U0,w1_trained;
	U.resize(30, vector<double>(10));
	U0.resize(30, vector<double>(10));
	w1_trained.resize(30, vector<double>(10));
	vector<double> B, B0,b0_trained;
	B.resize(30);
	B0.resize(30);
	b0_trained.resize(30);
	vector<double> C, C0, b1_trained;
	C.resize(10);
	C0.resize(10);
	b1_trained.resize(30);
	vector<double> Y,y0_test;
	Y.resize(30);
	y0_test.resize(30);
	vector<double> Z,y1_test;
	Z.resize(10);
	y1_test.resize(10);
	vector<double> SB,s0_test;
	SB.resize(30);
	s0_test.resize(30);
	vector<double> RC,s1_test;
	RC.resize(10);
	s1_test.resize(10);


	double e = 0.01;
	//initialize W
	for (int i = 0; i < 784; ++i) {
		for (int j = 0; j < 30; ++j) {
			W0[i][j] = gaussian_random();
		}

	}

	//initialize B
	for (int i = 0; i < 30; ++i) {
		B0[i] = gaussian_random();
	}

	//initialize U
	for (int i = 0; i < 30; ++i) {
		for (int j = 0; j < 10; ++j) {
			U0[i][j] = gaussian_random();
		}

	}
	//initialize C
	for (int i = 0; i < 10; ++i) {
		C0[i] = gaussian_random();
	}


	//initial Training
	cout << "Initial Training starts " << endl;
	for (int o = 0; o < data_considered; o++) {
            //Forward Propagation
            //Calculate Y
            for (int j=0; j<neurons_hidden_layer; j++) {
                for (int k=0; k<data_dimension; k++) {
                    SB[j] += W0[k][j]*train_data[o][k];
                }
            }
            for (int j=0; j<neurons_hidden_layer; j++)
                 SB[j] += B0[j];

            for(int j=0; j<neurons_hidden_layer; j++)
                Y[j] = tanh(SB[j]);


            //Calculate Z
            for (int j=0; j<labels_dimension; j++) {
                for (int k=0; k<neurons_hidden_layer; k++) {
                    RC[j] += U0[k][j]*Y[k];
                }
            }
            for (int j=0; j<labels_dimension; j++)
                 RC[j] += C0[j];

            for(int j=0; j<labels_dimension; j++)
                Z[j] = tanh(RC[j]);



            //Backpropagation starts
            //Update U
            for (int i = 0; i < 30; ++i) {
                for (int j = 0; j < 10; ++j) {
                    U[i][j] = U0[i][j] + e  * (Z[j] - target[o][j])*(Z[j]*Z[j] - 1)*Y[i];
                    //cout << U[i][j]<<" ";
                }
                //cout << endl;
            }

            //Update C
            for (int i = 0; i < 10; ++i) {
                C[i] = C0[i] - e * (Z[i] - target[o][i])*(1 - Z[i]*Z[i]);
                //   cout<<C[j]<<" ";//
            }
             //Uppdate B
            double b_Sum = 0.0;
            for (int i = 0; i < 30; ++i) {
                b_Sum=0.0;
                for (int j = 0; j < 10; ++j) {
                    b_Sum += (Z[j] - target[o][j])*(1 - Z[j]*Z[j]) * U0[i][j] *  (1 - Y[i]*Y[i]);
                }
                B[i] = B0[i] - e * b_Sum;
                //cout<<B[i]<<" ";
            }
             //Update W
            double w_Sum = 0.0;

            for (int k = 0; k < 784; ++k) {
                for (int n = 0; n < 30; ++n) {// for 30 neuron
                    w_Sum=0.0;
                    for (int i = 0; i < 10; ++i) {//10ta sum
                        w_Sum += (Z[i] - target[o][i])*(1.0 - Z[i]*Z[i]) * U0[n][i] * (1.0 - Y[n]*Y[n]) * train_data[o][k];

                    }
                    W[k][n] = W0[k][n] - e * w_Sum;
                    //cout<<W[k][n]<<" ";
                }
                //cout<<endl;
            }
            for (int j=0; j<labels_dimension; j++)
                outfile << Z[j] << ":" << target[o][j] << ":" << Z[j] - target[o][j] << ", ";
            outfile << endl;

            B0 = B;
            W0 = W;
            C0 = C;
            U0 = U;
            fill(SB.begin(), SB.end(), 0);
            fill(RC.begin(), RC.end(), 0);

        }

    // compute correct and wrong
    cout << "Initial Training Ends " << endl;

	//epoch
	for (int j = 0; j < epochs; j++) {
		cout << "BEGINING OF EPOCH " << j + 2 << endl;
		for (int o = 0; o < data_considered; o++) {
            //Forward Propagation
            //Calculate Y
            for (int j=0; j<neurons_hidden_layer; j++) {
                for (int k=0; k<data_dimension; k++) {
                    SB[j] += W0[k][j]*train_data[o][k];
                }
            }
            /*for (int j=0; j<labels_dimension; j++) {
                cout<<SB[j]<<" ";
            }
            cout<<endl;*/
            for (int j=0; j<neurons_hidden_layer; j++)
                 SB[j] += B0[j];

            for(int j=0; j<neurons_hidden_layer; j++)
                Y[j] = tanh(SB[j]);


            //Calculate Z
            for (int j=0; j<labels_dimension; j++) {
                for (int k=0; k<neurons_hidden_layer; k++) {
                    RC[j] += U0[k][j]*Y[k];
                }
            }

            for (int j=0; j<labels_dimension; j++)
                 RC[j] += C0[j];

            for(int j=0; j<labels_dimension; j++)
                Z[j] = tanh(RC[j]);



            //Backpropagation starts
            //Update U
            for (int i = 0; i < 30; ++i) {
                for (int j = 0; j < 10; ++j) {
                    U[i][j] = U0[i][j] + e  * (Z[j] - target[o][j])*(Z[j]*Z[j] - 1)*Y[i];
                    //cout << U[i][j]<<" ";
                }
                //cout << endl;
            }

            //Update C
            for (int i = 0; i < 10; ++i) {
                C[i] = C0[i] - e * (Z[i] - target[o][i])*(1 - Z[i]*Z[i]);
            }
            //for (int j=0; j<labels_dimension; j++) {
             //   cout<<C[j]<<" ";//
            //}
             //Uppdate B
            double b_Sum = 0.0;
            for (int i = 0; i < 30; ++i) {
                b_Sum=0.0;
                for (int j = 0; j < 10; ++j) {
                    b_Sum += (Z[j] - target[o][j])*(1 - Z[j]*Z[j]) * U0[i][j] *  (1 - Y[i]*Y[i]);
                }
                B[i] = B0[i] - e * b_Sum;
                //cout<<B[i]<<" ";
            }
             //Update W
            double w_Sum = 0.0;

            for (int k = 0; k < 784; ++k) {
                for (int n = 0; n < 30; ++n) {// for 30 neuron
                    w_Sum=0.0;
                    for (int i = 0; i < 10; ++i) {//10ta sum
                        w_Sum += (Z[i] - target[o][i])*(1.0 - Z[i]*Z[i]) * U0[n][i] * (1.0 - Y[n]*Y[n]) * train_data[o][k];

                    }
                    W[k][n] = W0[k][n] - e * w_Sum;
                    //cout<<W[k][n]<<" ";
                }
                //cout<<endl;
            }
            for (int j=0; j<labels_dimension; j++)
                outfile << Z[j] << ":" << target[o][j] << ":" << Z[j] - target[o][j] << ", ";
            outfile << endl;

            B0 = B;
            W0 = W;
            C0 = C;
            U0 = U;
            fill(SB.begin(), SB.end(), 0);
            //fill(Y.begin(), Y.end(), 0);
            fill(RC.begin(), RC.end(), 0);
            //fill(Z.begin(), Z.end(), 0);

        }
        cout<< "End of epoch "<<j+2<<endl;
        w0_trained = W;
        b0_trained = B;
        w1_trained = U;
        b1_trained = C;
        int accuracy=0;
        for (int i = 0; i < data_considered; i++){
            // FORWARD PROPAGATION
                for (int j=0; j<neurons_hidden_layer; j++) {
                    for (int k=0; k<data_dimension; k++) {
                        s0_test[j] += w0_trained[k][j]*train_data[i][k];
                    }
                }

                for (int j=0; j<neurons_hidden_layer; j++)
                     s0_test[j] += b0_trained[j];

                for(int j=0; j<neurons_hidden_layer; j++)
                    y0_test[j] = tanh(s0_test[j]);



                for (int j=0; j<labels_dimension; j++) {
                    for (int k=0; k<neurons_hidden_layer; k++) {
                        s1_test[j] += w1_trained[k][j]*y0_test[k];
                    }
                }

                for (int j=0; j<labels_dimension; j++)
                     s1_test[j] += b1_trained[j];

                for(int j=0; j<labels_dimension; j++)
                    y1_test[j] = tanh(s1_test[j]);

                double maxm = *max_element(y1_test.begin(), y1_test.end());
                int index=0;
                for (int k =0; k<10; k++){
                    if (y1_test[k]==maxm) index= k;
                }

                accuracy += train_labels[i] == index ? 1 : 0;
                fill(s0_test.begin(), s0_test.end(), 0);
                fill(y0_test.begin(), y0_test.end(), 0);
                fill(s1_test.begin(), s1_test.end(), 0);
                fill(y1_test.begin(), y1_test.end(), 0);
        }
        cout<<"number of correct = "<<accuracy<<endl;
        cout<<"number of incorrect = "<<data_considered-accuracy<<endl;
        accuracy = ((accuracy*1.0)/data_considered)*100;
        cout << "accuracy ="<< accuracy << "%" << endl;



	}
	cout<<"end of all epoch"<<endl;

	outfile.close();
    plot.close();
    save_trained_weights(W, B, U, C);


}

int main()
{
    allocate();
    read_MNIST_data_train();
    read_MNIST_labels_train();
    read_MNIST_data_test();
    read_MNIST_labels_test();

    train();


	return 0;
}
