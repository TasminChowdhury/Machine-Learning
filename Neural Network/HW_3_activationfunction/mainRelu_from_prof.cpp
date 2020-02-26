#include <iostream>
#include <fstream>
using std::ofstream;
#include <cmath>
using namespace std;
int main()
{
	ofstream outfile("file2.dat");
	int i,j,r1,r2;
	int X1[2000], X2[2000], T[2000],z1;
	int max = INT_MAX;
	float s1, s2, r, y1, y2, z, w11, w21,w12,w22,u1,u2,w110,w210,w120,w220,u10,u20,e,s1b,s2b,rb;
	float  b0,b,b1,b2,b10,b20;
	outfile << "INT_MAX = " <<max <<"  "<< INT_MAX << endl;
	outfile << "RAND_MAX= " << RAND_MAX << endl;
	w110 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "w11= " << w110 << endl;
	w210 = float(rand()) / float(RAND_MAX) -0.5;
	outfile << "w21= " << w210 << endl;
	w120 = float(rand()) / float(RAND_MAX) -0.5;
	outfile << "w12= " << w120 << endl;
	w220 = float(rand()) / float(RAND_MAX) -0.5;
	outfile << "w22= " << w220 << endl;
	u10 = float(rand()) / float(RAND_MAX) -0.5;
	outfile << "u1= " << u10 << endl;
	u20 = float(rand()) / float(RAND_MAX) -0.5;
	outfile << " u20= " << u20 << endl;
	b10 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "b1= " << b10 << endl;
	b20 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "b2= " << b20 << endl;
	b0 = float(rand()) / float(RAND_MAX) - 0.5;
	outfile << "b= " << b0 << endl;

	e = 0.01;
	
	for (i = 0; i < 2000; i++)
	{
		//cout <<  ((double)rand() / (RAND_MAX))  << endl;
		r1 = rand() % 2;
		X1[i] = r1;
		r2 = rand() % 2;
		X2[i] = r2;
		if (X1[i] == X2[i]) T[i] = 0;
		else T[i] = 1;
		outfile << X1[i] << " " <<  X2[i] << " " << T[i] << " ";
		s1 = w110*X1[i] + w210*X2[i];
		s2 = w120*X1[i] + w220*X2[i];
		s1b = s1 + b10;
		s2b = s2 + b20;
		if (s1b > 0)y1 = s1b;
		else y1 = 0;
		if (s2b > 0) y2 = s2b;
		else y2 = 0;
		r = u10*y1 + u20*y2;
		rb = r + b0;
		if (rb > 0)
		{
			z = rb;
			b = b0 - e*(z - T[i]);
			if (s1b > 0)
			{
				y1 = s1b;
				u1 = u10 - e*(z - T[i])*y1;
				b1 = b10 - e*(z - T[i])*u1;
				w11 = w110 - e*(z - T[i])*u1*X1[i];
				w12 = w120 - e*(z - T[i])*u1*X2[i];
			}
			else
			{
				y1 = 0;
				u1 = u10;
				b1 = b10;
				w11 = w110;
				w12 = w120;
			}
			if (s2b > 0)
			{
				y2 = s2b;
				u2 = u20 - e*(z - T[i])*y2;
				b2 = b20 - e*(z - T[i])*u2;
				w21 = w210 - e*(z - T[i])*u2*X1[i];
				w22 = w220 - e*(z - T[i])*u2*X2[i];

			}
			else
			{
				y2 = 0;
				u2 = u20;
				b2 = b20;
				w21 = w210;
				w22 = w220;
			}
		}
		else
		{
			z = 0;
			b = b0;
		}
		if (rb > 0)
		{
			z = rb;
			b = b0 - e*(z - T[i]);
			u1 = u10 - e*(z - T[i])*y1;
			u2 = u20 - e*(z - T[i])*y2;
			if (s1b > 0)
			{
				b1 = b10 - e*(z - T[i])*u1;
				w11 = w110 - e*(z - T[i])*u1*X1[i];
				w21 = w210 - e*(z - T[i])*u1*X2[i];
			}
			else
			{
				b1 = b10;
				w11 = w110;
				w21 = w210;

			}
			if (s2b > 0)
			{
				b2 = b20 - e*(z - T[i])*u2;
				w12 = w120 - e*(z - T[i])*u2*X1[i];
				w22 = w22 - e*(z - T[i])*u2*X2[i];
			}
			else
			{
				b2 = b20;
				w12 = w120;
				w22 = w220;
			}
		}
		else
		{
			z = 0;
			b = b0;
			b2 = b20;
			w12 = w120;
			w22 = w220;
			b1 = b10;
			w11 = w110;
			w21 = w210;
			u1 = u10;
			u2 = u20;
		}
		
		outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u1=" << u1 << " u2=" << u2 << endl;
		w110 = w11;
		w210 = w21;
		w120 = w12;
		w220 = w22;
		u10 = u1;
		u20 = u2;
		b0 = b;
		b10 = b1;
		b20 = b2;
		outfile << "s1=" << s1 << " s2=" << s2 << " y1=" << y1 << " y2=" << y2 << " z=" << z <<"  error = " << z - T[i] << endl; 
	} 
	for (j = 0; j < 50; j++)
	{
		outfile << "BEGINING OF EPOCH " << j + 2 << endl;
		for (i = 0; i < 2000; i++)
		{
			//cout <<  ((double)rand() / (RAND_MAX))  << endl;
		
			outfile << X1[i] << " " << X2[i] << " " << T[i] << " ";
			s1 = w110*X1[i] + w210*X2[i];
			s2 = w120*X1[i] + w220*X2[i];
			s1b = s1 + b10;
			s2b = s2 + b20;
			if (s1b > 0)y1 = s1b;
			else y1 = 0;
			if (s2b > 0) y2 = s2b;
			else y2 = 0;  
			r = u10*y1 + u20*y2;
			rb = r + b0;
				if (rb > 0)
				{
					z = rb;
					b = b0 - e*(z - T[i]);
					if (s1b > 0)
					{
						y1 = s1b;
						u1 = u10 - e*(z - T[i])*y1;
						b1 = b10 - e*(z - T[i])*u1;
						w11 = w110 - e*(z - T[i])*u1*X1[i];
						w12 = w120 - e*(z - T[i])*u1*X2[i];
					}
					else
					{
						y1 = 0;
						u1 = u10;
						b1 = b10;
						w11 = w110;
						w12 = w120;
					}
					if (s2b > 0)
					{
						y2 = s2b;
						u2 = u20 - e*(z - T[i])*y2;
						b2 = b20 - e*(z - T[i])*u2;
						w21 = w210 - e*(z - T[i])*u2*X1[i];
						w22 = w220 - e*(z - T[i])*u2*X2[i];

					}
					else
					{
						y2 = 0;
						u2 = u20;
						b2 = b20;
						w21 = w210;
						w22 = w220;
					}
				}
				else
				{
					z = 0;
					b = b0;
					b2 = b20;
					w12 = w120;
					w22 = w220;
					b1 = b10;
					w11 = w110;
					w21 = w210;
					u1 = u10;
					u2 = u20;
					
				}
			outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u1=" << u1 << " u2=" << u2<<"b = "<<b<<"b1= "<<b1<< " b2= " << b2 << endl;
			w110 = w11;
			w210 = w21;
			w120 = w12;
			w220 = w22;
			u10 = u1;
			u20 = u2;
			b0 = b;
			b10 = b1;
			b20 = b2;

			outfile << "s1=" << s1 << " s2=" << s2 << " y1=" << y1 << " y2=" << y2 << " z=" << z << "  error = " << z - T[i] << endl;
		}
		outfile << " End of Epoch " << j  << endl;
	}
	return 0;
}