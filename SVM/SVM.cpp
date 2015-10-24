#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <vector>
#include <queue>
#include <map>
#include <set>
#define INF 0x3f3f3f3f
#define LL long long
#define max(x, y) ((x > y) ? x : y)
#define min(x, y) ((x < y) ? x : y)
#define eq(x, y) (fabs(x-y) < 1e-8)
#define lt(x, y) (x < y && !eq(x, y))
#define gt(x, y) (x > y && !eq(x, y))
#define le(x, y) (x < y || eq(x, y))
#define ge(x, y) (x > y || eq(x, y))

#define N 269 									// Train Number
#define Xd 10 									// X's dimension
#define InputFile "heart_scale"					// Input File

using namespace std;

double TestX[ N ][ Xd ];
double TestY[ N ];

class SVM{

public:
	const static double C = 1.0; 				// Punishment Parameter
	const static double xi = 1e-2; 				// endure limit value
	const static int LimitIterationTime = 5; 	// MAX Iteration Times

	double b;									// b
	double alpha[ N ];							// alpha
	double Ks[ N ][ N ];						// cache for K(i, j)
	double Es[ N ];								// cache for E(i)
	int a1, a2;									// Index of alpha1 && alpha2

	SVM();
	void SMO_Train();							// Train
	double SVM_TEST();							// Test : return correct rate
	void InitialKs();							// Get Ks
	void InitialEs();							// Get Es
			
	inline double E(int i);						// error of Predict Fuction
	inline double G(int i);						// Predict Fuction
	inline double K(int i, int j); 				// kernel function (changable)

};

void INPUT();

/*************	MAIN.cpp  ***************/

int main()
{
	INPUT();
	SVM T;
	printf("Start SVM Train =====> \n");
	T.SMO_Train();
	printf("Start SVM Test =====> \n");
	printf("Accuracy Rate : %f\n", T.SVM_TEST());
	return 0;
}

/****************************************/

void INPUT()
{
	FILE* in = fopen(InputFile, "r");
	char s[1000];
	for(int i = 0; i < N; i ++)
	{
		fgets(s, 1000, in);
		if(s[0] == '+')
			TestY[i] = 1;
		else
			TestY[i] = -1;
		for(int j = 0, k = 0; j < strlen(s) && k < Xd; j ++)
			if(s[j] == ':')
			{
				TestX[i][k] = atof(&s[j+1]);
				k ++;
			}
	}
/*	for(int i = 0; i < N; i ++)
	{
		for(int j = 0; j < Xd; j ++)
			printf("%f ", TestX[i][j]);
	}*/
	fclose(in);
}

SVM::SVM()
{
	b = 0;
	a1 = 0;
	a2 = 0;
	memset(alpha, 0, sizeof(alpha));
	memset(Ks, 0, sizeof(Ks));
	memset(Es, 0, sizeof(Es));
}

void SVM::InitialKs()
{
	for(int i = 0; i < N; i ++)
		for(int j = 0; j < N; j ++)
			Ks[i][j] = K(i, j);
}

void SVM::InitialEs()
{
	for(int i = 0; i < N; i ++)
		Es[i] = E(i);
}

void SVM::SMO_Train()
{
	int iter = 0;
	while(iter < LimitIterationTime)
	{
		printf("iter: %d\n", iter);
		int alpha_change_num = 0;
		InitialKs();	// get Kij	(maybe put different place to improve effect)
		InitialEs();

		for(a1 = 0; a1 < N; a1 ++)
			// find alpha1 which disobey KKT condition
			if( (TestY[a1] * Es[a1] > xi && gt(alpha[a1], 0)) ||
				(TestY[a1] * Es[a1] < -xi && lt(alpha[a1], C)) )// ||
				// (eq(TestY[a1] * E[a1], 1.0) && lt(0, alpha[a1]) && lt(alpha[a1], C)) )
			{
				InitialEs();

				// find alpha2 which maximize(E2 - E1)
				a2 = 0;
				for(int i = 0; i < N; i ++)
					if(fabs(Es[a1] - Es[i]) > fabs(Es[a1] - Es[a2]))
						a2 = i;

				double oldalpha1 = alpha[a1];
				double oldalpha2 = alpha[a2];

				// update alpha2
				double eta = Ks[a1][a1] + Ks[a2][a2] - 2.0 * Ks[a1][a2];

				if(eta <= 0)										// ??????????
					continue;
				alpha[a2] += TestY[a2] * (Es[a1] - Es[a2]) / eta;

				// cutting alpha2
				double L, H;
				if(!eq(TestY[a1], TestY[a2]))
					L = max(0, oldalpha2 - oldalpha1), H = min(C, C + oldalpha2 - oldalpha1);
				else
					L = max(0, oldalpha2 + oldalpha1 - C), H = min(C, oldalpha2 + oldalpha1);

				if(alpha[a2] > H)
					alpha[a2] = H;
				if(alpha[a2] < L)
					alpha[a2] = L;

				if(fabs(oldalpha2 - alpha[a2]) < 1e-5)
					continue;

				// update alpha1
				alpha[a1] += TestY[a1] * TestY[a2] * (oldalpha2 - alpha[a2]);

				// get b1 & b2
				double b1 = b - Es[a1] - TestY[a1] * Ks[a1][a1] * (alpha[a1] - oldalpha1) - TestY[a2] * Ks[a2][a1] * (alpha[a2] - oldalpha2);
				double b2 = b - Es[a2] - TestY[a1] * Ks[a1][a2] * (alpha[a1] - oldalpha1) - TestY[a2] * Ks[a2][a2] * (alpha[a2] - oldalpha2);
				if(0 < alpha[a1] && alpha[a1] < C)
					b = b1;
				else if(0 < alpha[a2] && alpha[a2] < C)
					b = b2;
				else
					b = (b1 + b2) / 2;

				alpha_change_num ++;
			}

		if(alpha_change_num == 0)
			iter ++;
		else
			iter = 0;
	}
}

double SVM::SVM_TEST()
{
	double rate = 0.0;
	double correct = 0.0;

	InitialKs();
	for(int i = 0; i < N; i ++)
	{
		printf("%f %f\n", G(i), TestY[i]);
	}
	for(int i = 0; i < N; i ++)
	{
		if(G(i) * TestY[i] > 0)
			correct ++;
	}
	return correct / N;
}

inline double SVM::E(int i)				// error of Predict Fuction
{
	return G(i) - TestY[i];
}

inline double SVM::G(int i)				// Predict Fuction
{
	double y = b;
	for(int j = 0; j < N; j ++)
		y += alpha[j] * TestY[j] * Ks[i][j];
	return y;
}

inline double SVM::K(int i, int j)		// kernel function (changable)
{
	double y = 0.0;
	for(int k = 0; k < Xd; k ++)
		y += TestX[i][k] * TestX[j][k];
	return y;
}