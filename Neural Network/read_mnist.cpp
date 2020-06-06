#include "stdafx.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <conio.h>

using namespace std;
vector< vector<double> > arr;
int ReverseIntData(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNISTData()
{
	ifstream file;
	int NoOfImg = 10000, DataOfImg = 784;
	arr.resize(NoOfImg, vector<double>(DataOfImg));
	file.open("C:\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", ios::binary);

	if (file.is_open())
	{
		int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
		// Reading from file magic number, #of images, #of rows and #of columns
		//CvSize size;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseIntData(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseIntData(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseIntData(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseIntData(n_cols);
		//unsigned char arrshow[28][28];
		FILE *f;
		if (!f) {
			perror("File opening failed");
		}
		//storing only 10 characters
		for (int i = 0; i<10; ++i)
		{
			for (int row = 0; row<n_rows; ++row)
			{
				for (int column = 0; column<n_cols; ++column)
				{
					unsigned char flag = 0;
					file.read((char*)&flag, sizeof(flag));
					arr[i][(n_rows*row) + column] = flag;
					f = fopen("../image.png", "w");
					fprintf(f, "P3\n%d %d\n255\n", 28, 28);


					//printing those 10 characters only
					if (arr[i][(n_rows*row) + column] > 1) {
						//cout <<1/* arr[i][(n_rows*row) + column]*/ << " ";
						fprintf(f, "%d ", (int)arr[i][(n_rows*row) + column]);
					}
					else
						//cout << arr[i][(n_rows*row) + column]<< " ";
						fprintf(f, "%d ", (int)arr[i][(n_rows*row) + column]);
				}

				fprintf(f, "\n");
			}
		}
		fclose(f);
	}

	else {
		cout << "cant open the file";
	}
}

int main()
{
	ReadMNISTData();
	getchar();
	return 0;
}

