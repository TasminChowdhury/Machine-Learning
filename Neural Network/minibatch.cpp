

*******
#include <iostream>
#include <fstream>
using std::ofstream;
#include <cmath>
using namespace std;
int main()
{
    ofstream outfile("filebatch1.txt");
    int i,j,k,count;
    int X1[2000], X2[2000], Txor[2000],Tand[2000],X1b[2000],X2b[2000],Tb[2000],r1,r2;
    bool xb[2000];
    int max = INT_MAX;
    float s1, s2, m1,m2, y1, y2, z1,z2, w11, w21,w12,w22,u1,u2,w110,w210,w120,w220,u110,u210,u120,u220,e;
    float  c10,c20,c1,c2,b1,b2,b10,b20;
    float c1sum,c2sum, b1sum, b2sum, u11sum, u21sum,u12sum, u22sum, w11sum, w12sum, w21sum, w22sum;
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

    e = 0.1;



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

    }
    for(k = 0; k < 2000; k=k+10)
    {
        s1 = w110*X1[k] + w210*X2[k];
        s2 = w120*X1[k] + w220*X2[k];
        y1 = tanh(s1+b10);
        y2 = tanh(s2+b20);
        m1 = u110*y1 + u210*y2;
        m2 = u120 * y1 + u220 * y2;
        z1 = tanh(m1+c10);
        z2 =tanh(m2+c20);
        outfile << X1[k] << " " << X2[k] << " " << Txor[k] << " " << " error= " << z1 - Txor[k] << " ";
        outfile << X1[k] << " " << X2[k] << " " << Tand[k] << " " << " error= " << z1 - Tand[k] << " ";
        c1sum = 0;
        c2sum = 0;
        b1sum = 0;
        b2sum = 0;
        u11sum = 0;
        u21sum = 0;
        u12sum = 0;
        u22sum = 0;
        w11sum = 0.0;
        w21sum = 0.0;
        w22sum = 0.0;
        w12sum = 0.0;
        for (i = 0; i < 10; i++)
        {
            s1 = w110*X1[k + i] + w210*X2[k + i];
            s2 = w120*X1[k + i] + w220*X2[k + i];
            y1 = tanh(s1 + b10);
            y2 = tanh(s2 + b20);
            m1 = u110*y1 + u210*y2;

            z1 = tanh(m1 + c10);
            z2 = tanh(m2 + c20);
            c1sum = c1sum + (z1 - Txor[k + i])*(1.0 - z1*z1);
            c2sum = c2sum + (z2 - Tand[k + i])*(1.0 - z2*z2);

            u11sum = u11sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*y1;
            u21sum = u21sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*y2;
            u12sum = u12sum + (z2 - Tand[k + i])*(1.0 - z2*z2)*y1;
            u22sum = u22sum + (z2 - Tand[k + i])*(1.0 - z2*z2)*y2;

            b1sum = b1sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1) + (z2-Tand[k+i])* (1-z2*z2)*u120*(i-y1*y1);
            b2sum = b2sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2) + (z2-Tand[k+i])* (1-z2*z2)*u220*(i-y2*y2);

            w11sum = w11sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1)*X1[i+k];
            w21sum = w21sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1)*X2[i + k];
            w12sum = w12sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2)*X1[i + k];
            w22sum = w22sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2)*X2[i + k];
        }

        c1sum = c1sum / 10.0;
        c2sum = c2sum / 10.0;
        b1sum = b1sum / 10.0;
        b2sum = b2sum / 10.0;
        u11sum = u11sum / 10.0;
        u21sum = u21sum / 10.0;
        u12sum = u12sum / 10.0;
        u22sum = u22sum / 10.0;
        w11sum = w11sum / 10.0;
        w21sum = w21sum / 10.0;
        w12sum = w12sum / 10.0;
        w22sum = w22sum / 10.0;

        u11 = u110 - e*u11sum;
        u21 = u210 - e*u21sum;
        w11 = w110 - e* w11sum;
        w21 = w210 - e* w21sum;
        w12 = w120 - e* w12sum;
        w22 = w220 - e * w22sum;
        c1 = c10 - e* c1sum;
        c2 = c20 - e* c2sum;
        b1 = b10 - b1sum;
        b2 = b20 - b2sum;
        outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u1=" << u1 << " u2=" << u2 << endl;
        w110 = w11;
        w210 = w21;
        w120 = w12;
        w220 = w22;
        u110 = u1;
        u210 = u2;
        c10 = c1;
        b10 = b1;
        b20 = b2;
    }
    for (j = 0; j < 10; j++)//10 epoch
    {
        for (k = 0; k < 2000; k++)
        {
            xb[k] = 0;
            X1b[k] = 0;
        }
        count = 0;
        while (count  < 2000)
        {
            r1 = rand() % 2000;
            if ((xb[r1] == 0) && (X1b[r1] == 0))
            {
                X1b[r1] = X1[r1];
                X2b[r1] = X2[r1];
                Tb[r1] = Txor[r1];
                xb[r1] = 1;
                count++;
            }

        }
        for (k = 0; k < 2000; k++)
        {
            X1[k] = X1b[k];
            X2[k] = X2b[k];
            Txor[k] = Tb[k];
        }
        outfile << "BEGINING OF EPOCH " << j + 1 << endl;

        for (k = 0; k < 2000; k=k + 10)
        {
            s1 = w110*X1[k] + w210*X2[k];
            s2 = w120*X1[k] + w220*X2[k];
            y1 = tanh(s1 + b10);
            y2 = tanh(s2 + b20);
            m1 = u110*y1 + u210*y2;
            z1 = tanh(m1 + c10);
            outfile << X1[k] << "  " << X2[k] << " " << Txor[k] << " error = " << z1-Txor[k] << " ";
            c1sum = 0;
            b1sum = 0;
            b2sum = 0;
            u11sum = 0;
            u21sum = 0;
            w11sum = 0.0;
            w21sum = 0.0;
            w22sum = 0.0;
            w12sum = 0.0;
            for (i = 0; i < 10; i++)
            {
                s1 = w110*X1[k + i] + w210*X2[k + i];
                s2 = w120*X1[k + i] + w220*X2[k + i];
                y1 = tanh(s1 + b10);
                y2 = tanh(s2 + b20);
                m1 = u110*y1 + u210*y2;
                z1 = tanh(m1 + c10);
                c1sum = c1sum + (z1 - Txor[k + i])*(1.0 - z1*z1);
                u11sum = u11sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*y1;
                u21sum = u21sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*y2;
                b1sum = b1sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1);
                b2sum = b2sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2);
                w11sum = w11sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1)*X1[i + k];
                w21sum = w21sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u110*(1 - y1*y1)*X2[i + k];
                w12sum = w12sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2)*X1[i + k];
                w22sum = w22sum + (z1 - Txor[k + i])*(1.0 - z1*z1)*u210*(1 - y2*y2)*X2[i + k];
            }

            c1sum = c1sum / 10.0;
            b1sum = b1sum / 10.0;
            b2sum = b2sum / 10.0;
            u11sum = u11sum / 10.0;
            u21sum = u21sum / 10.0;
            w11sum = w11sum / 10.0;
            w21sum = w21sum / 10.0;
            w12sum = w12sum / 10.0;
            w22sum = w22sum / 10.0;

            u1 = u110 - e*u11sum;
            u2 = u210 - e*u21sum;
            w11 = w110 - e* w11sum;
            w21 = w210 - e* w21sum;
            w12 = w120 - e*w12sum;
            w22 = w220 - e * w22sum;
            c1 = c10 - e* c1sum;
            b1 = b10 - b1sum;
            b2 = b20 - b2sum;
            outfile << "w11= " << w11 << " w21= " << w21 << " w12=" << w12 << " w22=" << w22 << " u1=" << u1 << " u2=" << u2 << endl;
            w110 = w11;
            w210 = w21;
            w120 = w12;
            w220 = w22;
            u110 = u1;
            u210 = u2;
            c10 = c1;
            b10 = b1;
            b20 = b2;
        }

        outfile << " End of Epoch " << j+1  << endl;
    }
    return 0;
}

