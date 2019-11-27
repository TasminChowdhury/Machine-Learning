#include <iostream>
#include <fstream>
using std::ofstream;
#include <cmath>
using namespace std;
int main()
{
    ofstream outfile("filedoublesigmoid.txt");
    int i,j,r1,r2;
    int X1[2000], X2[2000], T1[2000],T2[2000];
    int max = INT_MAX;
    float s1, s2,r4,r3, y1, y2, z1,z2, w11, w21,w12,w22,u11,u21,u12,u22,w110,w210,w120,w220,u110,u210,u120,u220,e;
    float  b0,b,b1,b2,b10,b20,b40,b4;
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
    u110 = float(rand()) / float(RAND_MAX) -0.5;
    outfile << "u11= " << u110 << endl;
    u210 = float(rand()) / float(RAND_MAX) -0.5;
    outfile << " u210= " << u210 << endl;
    u120 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << "u21= " << u210 << endl;
    u220 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << " u220= " << u220 << endl;
    b10 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << "b1= " << b10 << endl;
    b20 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << "b2= " << b20 << endl;
    b0 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << "b= " << b0 << endl;
    b40 = float(rand()) / float(RAND_MAX) - 0.5;
    outfile << "b4= " << b40 << endl;

    e = 0.2;



    for (i = 0; i < 2000; i++)
    {
        //cout <<  ((double)rand() / (RAND_MAX))  << endl;
        r1 = rand() % 2;
        X1[i] = r1;
        r2 = rand() % 2;
        X2[i] = r2;
        if (X1[i] == X2[i]) T1[i] = 0;
        else T1[i] = 1;
        outfile << X1[i] << " " <<  X2[i] << " " << T1[i] << " ";

        if ((X1[i] == 1) && (X2[i] == 1))T2[i] = 1;
        else T2[i] = 0;
        outfile << X1[i] << " " << X2[i] << " " << T2[i] << " ";
        s1 = w110*X1[i] + w210*X2[i];
        s2 = w120*X1[i] + w220*X2[i];
        y1 = (2.0 / (1.0 + exp(- s1 + b10)))-1;
        y2 = (2.0 / (1.0 + exp(-s2 + b20)))-1;
        r3 = u110*y1 + u210*y2;
        r4 = u120*y1 + u220*y2;
        z1 = (2.0 / (1.0 + exp(- r3 + b0)))-1;
        z2 = (2.0 / (1.0 + exp(-r4 + b40)))-1;
        u11 = u110 + e*0.5*(z1 - T1[i])*(z1*z1 - 1)*y1;
        u21 = u210 + e*0.5*(z1 - T1[i])*(z1*z1 - 1)*y2;
        u12 = u120 + e*0.5*(z2 - T2[i])*(z2*z2 - 1)*y1;
        u22 = u220 + e*0.5*(z2 - T2[i])*(z2*z2 - 1)*y2;
        w11 = w110 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u110 +(z2-T2[i])*(1.0-z2*z2)*0.5*u120)*(1.0-y1*y1)*0.5*X1[i];
        w21 = w210 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u110 +(z2-T2[i])*(1.0-z2*z2)*0.5*u120)*(1.0-y1*y1)*0.5*X2[i];
        w12 = w120 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u210 +(z2-T2[i])*(1.0-z2*z2)*0.5*u220)*(1.0-y2*y2)*0.5*X1[i];
        w22 = w220 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u210 +(z2-T2[i])*(1.0-z2*z2)*0.5*u220)*(1.0-y2*y2)*0.5*X2[i];
        b = b0 + e*(z1 - T1[i])*(1-z1*z1)*0.5;
        b4 = b40 + e*(z2 - T2[i])*(1 - z2*z2)*0.5;
        b1 = b10 + e*((z1 - T1[i])*(1.0 - z1*z1)*0.5*u110 + (z2 - T2[i])*(1.0 - z2*z2)*0.5*u120)*0.5*(1.0-y1*y1);
        b2 = b20 + e*((z1 - T1[i])*(1.0 - z1*z1)*0.5*u210 + (z2 - T2[i])*(1.0 - z2*z2)*0.5*u220)*0.5*(1.0-y2*y2);
        outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 <<" u11=" << u11 << " u21=" << u21 <<" u22= " << u22 << endl;
        w110 = w11;
        w210 = w21;
        w120 = w12;
        w220 = w22;
        u110 = u11;
        u210 = u21;
        u120 = u12;
        u220 = u22;
        b0 = b;
        b10 = b1;
        b20 = b2;
        b40 = b4;
        outfile  << " z1=" << z1 <<"  error1 = " << z1 - T1[i] << " z2=" <<z2<<" error2="<< z2-T2[i]<<  endl;
    }
    for (j = 0; j < 10; j++)
    {
        outfile << "BEGINING OF EPOCH " << j + 2 << endl;
        for (i = 0; i < 2000; i++)
        {
            outfile << X1[i] << " " << X2[i] << " " << T1[i] << " ";
            outfile << X1[i] << " " << X2[i] << " " << T2[i] << " ";
            s1 = w110*X1[i] + w210*X2[i];
            s2 = w120*X1[i] + w220*X2[i];

            y1 = (2.0 / (1.0 + exp(- s1 + b10)))-1;
            y2 = (2.0 / (1.0 + exp(-s2 + b20)))-1;
            r3 = u110*y1 + u210*y2;
            r4 = u120*y1 + u220*y2;
            z1 = (2.0 / (1.0 + exp(- r3 + b0)))-1;
            z2 = (2.0 / (1.0 + exp(-r4 + b40)))-1;
            u11 = u110 + e*0.5*(z1 - T1[i])*(z1*z1 - 1)*y1;
            u21 = u210 + e*0.5*(z1 - T1[i])*(z1*z1 - 1)*y2;
            u12 = u120 + e*0.5*(z2 - T2[i])*(z2*z2 - 1)*y1;
            u22 = u220 + e*0.5*(z2 - T2[i])*(z2*z2 - 1)*y2;
            w11 = w110 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u110 +(z2-T2[i])*(1.0-z2*z2)*0.5*u120)*(1.0-y1*y1)*0.5*X1[i];
            w21 = w210 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u110 +(z2-T2[i])*(1.0-z2*z2)*0.5*u120)*(1.0-y1*y1)*0.5*X2[i];
            w12 = w120 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u210 +(z2-T2[i])*(1.0-z2*z2)*0.5*u220)*(1.0-y2*y2)*0.5*X1[i];
            w22 = w220 - e*((1.0 - z1*z1)*(z1 - T1[i])*0.5*u210 +(z2-T2[i])*(1.0-z2*z2)*0.5*u220)*(1.0-y2*y2)*0.5*X2[i];
            b = b0 + e*(z1 - T1[i])*(1-z1*z1)*0.5;
            b4 = b40 + e*(z2 - T2[i])*(1 - z2*z2)*0.5;
            b1 = b10 + e*((z1 - T1[i])*(1.0 - z1*z1)*0.5*u110 + (z2 - T2[i])*(1.0 - z2*z2)*0.5*u120)*0.5*(1.0-y1*y1);
            b2 = b20 + e*((z1 - T1[i])*(1.0 - z1*z1)*0.5*u210 + (z2 - T2[i])*(1.0 - z2*z2)*0.5*u220)*0.5*(1.0-y2*y2);
            outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 <<  " u11=" << u11 << " u21=" << u21 << " u22= " << u22 << endl;
            w110 = w11;
            w210 = w21;
            w120 = w12;
            w220 = w22;
            u110 = u11;
            u210 = u21;
            u120 = u12;
            u220 = u22;
            b0 = b;
            b10 = b1;
            b20 = b2;
            b40 = b4;

            outfile << "s1=" << s1 << " s2=" << s2 << " y1=" << y1 << " y2=" << y2 << " z1=" << z1 << "  error1 = " << z1 - T1[i] << " z2=" << z2 << " error2=" << z2 - T2[i] << endl;
        }


        outfile << " End of Epoch " << j + 1 << endl;
    }

    return 0;
}