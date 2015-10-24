#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define Num_X 784+1
#define Num_Y 10
#define Num_M 25 //中间层
#define Num_Train 60000
#define Num_Test 10000
#define LoopTime 200
#define InitialMAX 255 // 用于初始化X, 将X缩小到[0,1]之间
#define pi 3.1415926535897932

#define File_TrainX "mnist_uint8_trainX.txt"
#define File_TrainY "mnist_uint8_trainY.txt"
#define File_TestX "mnist_uint8_testX.txt"
#define File_TestY "mnist_uint8_testY.txt"

using namespace std;

double TrainX[Num_X], TrainY[Num_Y];
double TestX[Num_X], TestY[Num_Y];
FILE *inX, *inY, *inMW, *outMW;
double TotalJ;

class ANN
{
	private:
		const static double rate = 0.1; 	// 学习速率
		const static double addrate = 0.1;	// 附加冲量率
	
	public:
		double m[Num_X][Num_M];
		double w[Num_M][Num_Y];
		double deltam[Num_X][Num_M];
		double deltaw[Num_M][Num_Y];

		double Mnet[Num_M];
		double Mid[Num_M];

		double Ynet[Num_Y];
		double Expect[Num_Y];

		double dJ_dMid[Num_M]; 				// J 到 中间层 的偏导
		double J;
	
		ANN(); 								// 初始化
		void getExpect(double []); 			// 获得期望值 Expect[]
		void Gradient_Descent(); 			// 梯度下降
		bool Predict(); 					// 返回一次测试是否正确

		inline double arctan(double y){ return atan(y)/pi+0.5; }
		inline double darctan(double y){ return 1.0/pi/(1.0 + y*y); }
};

void Train(ANN *);							// 训练
void Test(ANN *);							// 测试
void ReadMW(ANN *);							// 读入权值
void WriteMW(ANN *);						// 保存权值

/*************** 主程序 **************/

int main() 
{
	ANN *T = new ANN();
	ReadMW(T);
	for(int k = 0; k < LoopTime; k ++)
	{
		TotalJ = 0.0;
		printf("Times %d :\n", k);
		Train(T);
		Test(T);
	}
	WriteMW(T);
	return 0;
}

/*************************************/

void Train(ANN *T)
{
	inX = fopen(File_TrainX, "r");
	inY = fopen(File_TrainY, "r");
	for(int i = 0 ; i < Num_Train; i ++)
	{
		for(int j = 0; j < Num_X; j ++)
			if(j == 0)
				TrainX[j] = 1.0;
			else
			{
				fscanf(inX, "%lf", &TrainX[j]);
				TrainX[j] /= InitialMAX;
			}

		for(int j = 0; j < Num_Y; j ++)
		{
			fscanf(inY, "%lf", &TrainY[j]);
			TrainY[j] = 0.1 + 0.8*TrainY[j];
		}

		T->Gradient_Descent();
	}
	fclose(inX);
	fclose(inY);
}

void Test(ANN *T)
{
	TotalJ = 0.0;

	inX = fopen(File_TestX, "r");
	inY = fopen(File_TestY, "r");

	int correct = 0;
	for(int i = 0; i < Num_Test; i ++)
	{
		for(int j = 0; j < Num_X; j ++)
			if(j == 0)
				TestX[j] = 1.0;
			else
			{
				fscanf(inX, "%lf", &TestX[j]);
				TestX[j] /= InitialMAX;
			}

		for(int j = 0; j < Num_Y; j ++)
			fscanf(inY, "%lf", &TestY[j]);

		if(T->Predict())
			correct ++;
	}

	printf("	Accuracy rate : %.2f\n", (double)correct / Num_Test);
	printf("	Total J : %f\n", TotalJ);

	fclose(inX);
	fclose(inY);
}

ANN::ANN()
{
	J = 0.0;
	memset(m, 0, sizeof(m));
	memset(w, 0, sizeof(w));
	memset(deltaw, 0, sizeof(deltaw));
	memset(deltam, 0, sizeof(deltam));
	memset(Expect, 0, sizeof(Expect));
	memset(Mnet, 0, sizeof(Mnet));
	memset(Ynet, 0, sizeof(Ynet));
	memset(Mid, 0, sizeof(Mid));
	memset(dJ_dMid, 0, sizeof(dJ_dMid));
	for(int i = 0; i < Num_X; i ++)
		for(int j = 0; j < Num_M; j ++)
			m[i][j] = (double)rand()/10.0/RAND_MAX-0.05;
	for(int i = 0; i < Num_M; i ++)
		for(int j = 0; j < Num_Y; j ++)
			w[i][j] = (double)rand()/10.0/RAND_MAX-0.05;
}

void ANN::getExpect(double X[])
{
	for(int j = 0; j < Num_M; j ++)
	{
		double y = 0.0;
		for(int i = 0; i < Num_X; i ++)
			y += X[i] * m[i][j];
		Mid[j] = arctan(Mnet[j] = y);
	}

	for(int j = 0; j < Num_Y; j ++)
	{
		double y = 0.0;
		for(int i = 0; i < Num_M; i ++)
			y += Mid[i] * w[i][j];
		Expect[j] = arctan(Ynet[j] = y);
	}
}

void ANN::Gradient_Descent()
{
	getExpect(TrainX);

	memset(dJ_dMid, 0, sizeof(dJ_dMid));

	for(int j = 0; j < Num_Y; j ++)
		for(int i = 0; i < Num_M; i ++)
		{
			double dJ_dwnet = (TrainY[j] - Expect[j]) * darctan(Ynet[j]);
			dJ_dMid[i] += dJ_dwnet * w[i][j]; // dJ/dM = sigama(dJ/dwnet * dwnet/dMid)
			w[i][j] += rate * dJ_dwnet * Mid[i] + addrate * deltaw[i][j];
			deltaw[i][j] = rate * dJ_dwnet * Mid[i];
		}

	for(int j = 0; j < Num_M; j ++)
		for(int i = 0; i < Num_X; i ++)
		{
			double dMid_dmnet = darctan(Mnet[j]);
			m[i][j] += rate * dJ_dMid[j] * dMid_dmnet * TrainX[i] + addrate * deltam[i][j];
			deltam[i][j] = rate * dJ_dMid[j] * dMid_dmnet * TrainX[i];
		}
}

bool ANN::Predict()
{
	getExpect(TestX);
	double max = 0;
	int maxj = 0;
	for(int j = 0; j < Num_Y; j ++)
	{
		if(max < Expect[j])
			max = Expect[j], maxj = j;
		TotalJ += (Expect[j] - TestY[j])*(Expect[j] - TestY[j])/2;
	}

	if(fabs(TestY[maxj] - 1.0) < 0.01)
		return 1;
	return 0;
}

void ReadMW(ANN *T)
{
	inMW = fopen("outMW.txt", "r");
	for(int i = 0; i < Num_X; i ++)
		for(int j = 0; j < Num_M; j ++)
			fscanf(inMW, "%lf", &T->m[i][j]);
	for(int i = 0; i < Num_M; i ++)
		for(int j = 0; j < Num_Y; j ++)
			fscanf(inMW, "%lf", &T->w[i][j]);
	fclose(inMW);
}

void WriteMW(ANN *T)
{
	outMW = fopen("outMW.txt", "w");
	for(int i = 0; i < Num_X; i ++)
		for(int j = 0; j < Num_M; j ++)
			fprintf(outMW, "%.15f ", T->m[i][j]);
	for(int i = 0; i < Num_M; i ++)
		for(int j = 0; j < Num_Y; j ++)
			fprintf(outMW, "%.15f ", T->w[i][j]);
	fclose(outMW);
}