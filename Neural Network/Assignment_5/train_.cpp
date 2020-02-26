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
int feature_map_total = 6;
int max_pooling_window = 2;
int sliding_window = 5;
int size_of_feature_map = 24; //24*24
int down_sampled_image_size = 12; // 12*12

vector < vector< vector<double> > > train_data;
vector <int> train_labels;
vector < vector< vector<double> > > test_data;
vector <int> test_labels;
vector < vector<int> > target; // vectorized train labels

int data_considered=1;
int epochs= 1;

void allocate(){
    train_data.resize(number_of_training_data, vector< vector<double> >(28, vector<double>(28)));
    train_labels.resize(number_of_training_data);
    test_data.resize(number_of_test_data,vector< vector<double> >(28, vector<double>(28)));
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

        for(int i=0; i<10; i++)
        {
            for(int r=0; r<number_of_rows; r++)
            {
                for(int c=0; c<number_of_columns; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    train_data[i][r][c] = (float) (temp*1.0)/255;
                    cout<<train_data[i][r][c]<<" ";
                }
            }
            cout<<endl;
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

        for(int i=0; i<10; i++)
        {
            for(int r=0; r<number_of_rows; r++)
            {
                for(int c=0; c<number_of_columns; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    test_data[i][r][c] = (float) (temp*1.0)/255;
                    cout<<test_data[i][r][c]<<" ";
                }
            }
            cout<<endl;
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
    cout<< "----------train starting-----------"<<endl;
    ofstream outfile("file_log.txt");
    //ofstream plot("plot.csv");
	vector < vector< vector<double> > > W, W0,w0_trained;
	W.resize(feature_map_total, vector< vector<double> >(sliding_window, vector<double>(sliding_window)));
	W0.resize(feature_map_total, vector< vector<double> >(sliding_window, vector<double>(sliding_window)));
	//w0_trained.resize(feature_map_total, vector< vector<double> >(sliding_window, vector<double>(sliding_window)));

	vector<double> B, B0,b0_trained;
	B.resize(feature_map_total);
	B0.resize(feature_map_total);
	//b0_trained.resize(feature_map_total);

    vector < vector< vector<double> > > s0, y0, s1;
	s0.resize(feature_map_total, vector< vector<double> >(size_of_feature_map, vector<double>(size_of_feature_map)));
	y0.resize(feature_map_total, vector< vector<double> >(size_of_feature_map, vector<double>(size_of_feature_map)));
	s1.resize(feature_map_total, vector< vector<double> >(down_sampled_image_size, vector<double>(down_sampled_image_size)));


	double e = 0.01;
	//initialize W0
	for (int i = 0; i < feature_map_total; ++i) {
		for (int j = 0; j < sliding_window; ++j) {
		    for (int k = 0; k < sliding_window; ++k) {
                W0[i][j][k] = gaussian_random();
		    }
		}

	}

	//initialize B0
	for (int i = 0; i < feature_map_total; ++i) {
		B0[i] = gaussian_random();
	}

	//initial Training
	cout << "-------Initial Training starts-------" << endl;
	for (int o = 0; o < data_considered; o++) {
            //Forward Propagation
            // feature maps
            for (int i=0; i<feature_map_total; i++) {
                for (int j=0; j<size_of_feature_map; j++) {
                    for (int k=0; k<size_of_feature_map; k++) {

                        for (int m=0; m<sliding_window; m++) {
                            for (int n=0; n<sliding_window; n++) {
                                s0[i][j][k] += W0[i][m][n]* train_data[o][m][n];
                            }
                        }
                        s0[i][j][k] += B0[i];
                        y0[i][j][k] = tanh(s0[i][j][k
                        cout<<s0[i][j][k]<<endl;
                        cout<<y0[i][j][k]<<endl;
                    }

                }
            }
	}


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
