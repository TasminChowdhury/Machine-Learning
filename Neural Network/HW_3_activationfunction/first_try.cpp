#include <iostream>

#include <fstream>

using std::ofstream;

#include <cmath>

using namespace std;

int main()

{



	//Declarations and initialization of parameters

	ofstream outfile("file.dat");

	int i, j, r1, r2;

	int X1[2000], X2[2000], Txor[2000]/*z1*/,Tand[2000];

	int max = INT_MAX;

	float s1, s2, m1,m2, y1, y2, z1,z2, w11, w21, w12, w22, u11, u21,u12,u22, w110, w210, w120, w220, u110, u210,u120,u220, e;

	float  u3, c10, c1,c20,c2, b1, b2, b10, b20;

	outfile << "INT_MAX = " << max << "  " << INT_MAX << endl;
	outfile << "RAND_MAX= " << RAND_MAX << endl;

	w110 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "w11= " << w110 << endl;

	w210 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "w21= " << w210 << endl;

	w120 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "w12= " << w120 << endl;

	w220 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "w22= " << w220 << endl;

	u110 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "u11= " << u110 << endl;

	u210 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << " u21= " << u210 << endl;

	u120 = float(rand()) / float(RAND_MAX) - 0.5;

	outfile << "u12= " << u120 << endl;

	u220 = float(rand()) / float(RAND_MAX) - 0.5;

	outfile << " u22= " << u220 << endl;

	b10 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "b1= " << b10 << endl;

	b20 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "b2= " << b20 << endl;

	c10 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "c1= " << c10 << endl;

	c20 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "c2= " << c20 << endl;

	// Superparameter eta
	e = 0.1;





	//    Initial trainining of the NN

	for (i = 0; i < 2000; i++)

	{

		//cout <<  ((double)rand() / (RAND_MAX))  << endl;

		X1[i] = rand() % 2;
		X2[i] = rand() % 2;

		if (X1[i] == X2[i]) Txor[i] = 0;
		else Txor[i] = 1;
		outfile << X1[i] << " " << X2[i] << " " << Txor[i] << " ";

		if (X1[i] && X2[i]) Tand[i] = 1;
		else Tand[i] = 0;
		outfile << X1[i] << " " << X2[i] << " " << Tand[i] << " ";

		s1 = w110 * X1[i] + w210 * X2[i];
		s2 = w120 * X1[i] + w220 * X2[i];
		y1 = tanh(s1 + b10);
		y2 = tanh(s2 + b20);

		m1 = u110 * y1 + u210 * y2;
		m2 = u120 * y1 + u220 * y2;


		z1 = tanh(m1 + c10);
		z2 = tanh(m2 + c20);

		u11 = u110 - e * (z1 - Txor[i])*(1 - z1 * z1)*y1;
		u21 = u210 - e * (z1 - Txor[i])*(1 - z1 * z1)*y2;
		u12 = u120 - e * (z2 - Tand[i])*(1 - z2 * z2)*y1;
		u22 = u220 - e * (z2 - Txor[i])*(1 - z2 * z2)*y2;

		//w11 = w110 - e * (1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X1[i];
		w11 = w110 - e * ((1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X1[i] + (1.0 - z2 * z2)*(1.0 - y1 * y1)*(z2 - Tand[i])*u12*X1[i]);

		//w21 = w210 - e * (1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X2[i];
	    w21 = w210 - e * ((1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X2[i] + (1.0 - z2 * z2)*(1.0 - y1 * y1)*(z2 - Tand[i])*u12*X2[i]);
		//w12 = w120 - e * (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X1[i];
        w12 = w120 - e * ((1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X1[i] + (1.0 - z2 * z2)*(1.0 - y2 * y2)*(z2 - Tand[i])*u22*X1[i]);

		//w22 = w220 - e * (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X2[i];
		w22 = w220 - e *( (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X2[i]+(1.0 - z2 * z2)*(1.0 - y2 * y2)*(z2 - Tand[i])*u22*X2[i] );
		c1 = c10 - e * (z1 - Txor[i])*(1 - z1 * z1);
		c2 = c20 - e * (z2 - Tand[i])*(1 - z2 * z2);

		//b1 = b10 - e * (z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y1 * y1)*u11;
		b1 = b10 - e * ((z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y1 * y1)*u11 + (z2 - Tand[i])*(1.0 - z2 * z2)*(1.0 - y1 * y1)*u12);
		//b2 = b20 - e * (z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y2 * y2)*u21;
		b2 = b20 - e * ((z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y2 * y2)*u21 + (z2 - Tand[i])*(1.0 - z2 * z2)*(1.0 - y2 * y2)*u22);
		outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u11=" << u11 << " u21=" << u21 << endl;

		w110 = w11;
		w210 = w21;
		w120 = w12;
		w220 = w22;
		u110 = u11;
		u210 = u21;
		u120 = u12;
		u220 = u22;
		c10 = c1;
		c20=c2;
		b10 = b1;
		b20 = b2;

		outfile << "s1=" << s1 << " s2=" << s2 << " y1=" << y1 << " y2=" << y2 << " z=" << z1 << "  error = " << z1 - Txor[i] << endl;

	}



	// Only ten epochs are needed to finish the training

	for (j = 0; j < 10; j++)

	{

		outfile << "BEGINING OF EPOCH " << j + 1 << endl;

		for (i = 0; i < 2000; i++)

		{

			outfile << X1[i] << " " << X2[i] << " " << Txor[i] << " ";

			s1 = w110 * X1[i] + w210 * X2[i];

			s2 = w120 * X1[i] + w220 * X2[i];

			y1 = tanh(s1 + b10);

			y2 = tanh(s2 + b20);

			m1 = u110 * y1 + u210 * y2;
			m2 = u120 * y1 + u220 * y2;


			z1 = tanh(m1 + c10);
			z2 = tanh(m2 + c20);

			u11 = u110 - e * (z1 - Txor[i])*(1 - z1 * z1)*y1;
			u21 = u210 - e * (z1 - Txor[i])*(1 - z1 * z1)*y2;
			u12 = u120 - e * (z2 - Tand[i])*(1 - z2 * z2)*y1;
			u22 = u220 - e * (z2 - Txor[i])*(1 - z2 * z2)*y2;

			//w11 = w110 - e * (1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X1[i];
			w11 = w110 - e * ((1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X1[i] + (1.0 - z2 * z2)*(1.0 - y1 * y1)*(z2 - Tand[i])*u12*X1[i]);

			//w21 = w210 - e * (1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X2[i];
			 w21 = w210 - e * ((1.0 - z1 * z1)*(1.0 - y1 * y1)*(z1 - Txor[i])*u11*X2[i] + (1.0 - z2 * z2)*(1.0 - y1 * y1)*(z2 - Tand[i])*u12*X2[i]);
			//w12 = w120 - e * (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X1[i];
			w12 = w120 - e * ((1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X1[i] + (1.0 - z2 * z2)*(1.0 - y2 * y2)*(z2 - Tand[i])*u22*X1[i]);

			//w22 = w220 - e * (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X2[i];
			w22 = w220 - e *( (1.0 - z1 * z1)*(1.0 - y2 * y2)*(z1 - Txor[i])*u21*X2[i]+(1.0 - z2 * z2)*(1.0 - y2 * y2)*(z2 - Tand[i])*u22*X2[i] );
			c1 = c10 - e * (z1 - Txor[i])*(1 - z1 * z1);
			c2 = c20 - e * (z2 - Tand[i])*(1 - z2 * z2);

			//b1 = b10 - e * (z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y1 * y1)*u11;
			b1 = b10 - e * ((z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y1 * y1)*u11 + (z2 - Tand[i])*(1.0 - z2 * z2)*(1.0 - y1 * y1)*u12);
			//b2 = b20 - e * (z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y2 * y2)*u21;
			b2 = b20 - e * ((z1 - Txor[i])*(1.0 - z1 * z1)*(1.0 - y2 * y2)*u21 + (z2 - Tand[i])*(1.0 - z2 * z2)*(1.0 - y2 * y2)*u22);

			outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u11=" << u11 << " u21=" << u21 << "c1 = " << c1 << "b1= " << b1 << " b2= " << b2 << endl;

			w110 = w11;
			w210 = w21;
			w120 = w12;
			w220 = w22;
			u110 = u11;
			u210 = u21;
			u120 = u12;
			u220 = u22;
			c10 = c1;
			c20=c2;
			b10 = b1;
			b20 = b2;

			outfile << "s1=" << s1 << " s2=" << s2 << " y1=" << y1 << " y2=" << y2 << " z1=" << z1 << "  error = " << z1 - Txor[i] << endl;

		}

		outfile << " End of Epoch " << j + 1 << endl;

	}

	return 0;

}

