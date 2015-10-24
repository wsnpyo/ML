#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define Num_X 784+1
#define Num_Y 10
#define Num_Train 60000
#define Num_Test 10000
#define LoopTime 100
#define InitialMAX 500

#define File_TrainX "mnist_uint8_trainX.txt"
#define File_TrainY "mnist_uint8_trainY.txt"
#define File_TestX "mnist_uint8_testX.txt"
#define File_TestY "mnist_uint8_testY.txt"

using namespace std;

double TrainX[Num_X], TrainY[Num_Y];
double TestX[Num_X], TestY[Num_Y];
FILE *inX, *inY, *inW, *outW;
double TotalJ;

class ANN
{
	private:
		const static double rate = 0.3;
	
	public:
		double w[Num_X][Num_Y];
		double Expect[Num_Y];
	
		ANN(); // 初始化
		inline double sigmoid(double y){ return 1/(1+exp(-y)); } 
		void getExpect(double []); // 获得所有期望值
		void Gradient_Descent(); // 梯度下降
		bool Predict(); // 返回测试是否正确
};

void Train(ANN &);	// 训练
void Test(ANN &);	// 测试
void ReadW(ANN &);	// 读入权值
void WriteW(ANN &);	// 保存权值

int main()
{
	srand(time(NULL));
	ANN T;
	// ReadW(T);

	for(int k = 0; k < LoopTime; k ++)
	{
		printf("Times : %d =====> ", k);
		Train(T);
		Test(T);
	}

	WriteW(T);
	return 0;
}

void Train(ANN &T)
{
	inX = fopen(File_TrainX, "r");
	inY = fopen(File_TrainY, "r");
	for(int i = 0; i < Num_Train; i ++)
	{
		TrainX[0] = 1.0;
		for(int j = 1; j < Num_X; j ++)
		{
			fscanf(inX, "%lf", &TrainX[j]);
			TrainX[j] /= InitialMAX;
		}

		for(int j = 0; j < Num_Y; j ++)
		{
			fscanf(inY, "%lf", &TrainY[j]);
			// TrainY[j] = 0.1 + 0.8*TrainY[j];
		}

		T.Gradient_Descent();
	}
	fclose(inX);
	fclose(inY);
}

void Test(ANN &T)
{
	TotalJ = 0.0;
	
	inX = fopen(File_TestX, "r");
	inY = fopen(File_TestY, "r");

	int correct = 0;
	for(int i = 0; i < Num_Test; i ++)
	{
		TestX[0] = 1.0;
		for(int j = 1; j < Num_X; j ++)
		{
			fscanf(inX, "%lf", &TestX[j]);
			TestX[j] /= InitialMAX;
		}

		for(int j = 0; j < Num_Y; j ++)
			fscanf(inY, "%lf", &TestY[j]);

		if(T.Predict())
			correct ++;
	}

	printf("correct number : %d\n", correct);
	printf("Total J : %f\n", TotalJ);
	fclose(inX);
	fclose(inY);
}

ANN::ANN(){
	memset(Expect, 0, sizeof(Expect));
	memset(w, 0, sizeof(w));
	// for(int i = 0; i < Num_X; i ++)
	// 	for(int j = 0; j < Num_Y; j ++)
	// 		w[i][j] = (double)rand()/100.0/RAND_MAX-0.005;
}

void ANN::getExpect(double X[])
{
	for(int j = 0; j < Num_Y; j ++)
	{
		double y = 0.0;
		for(int i = 0; i < Num_X; i ++)
			y += w[i][j] * X[i];
		Expect[j] = sigmoid(y);
	}
}

void ANN::Gradient_Descent()
{
	getExpect(TrainX);
	for(int j = 0; j < Num_Y; j ++)
		for(int i = 0; i < Num_X; i ++)
		{
			double addi = rate * Expect[j] * (1.0 - Expect[j]) * (TrainY[j] - Expect[j]) * TrainX[i];
			w[i][j] += addi;
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
		{
			max = Expect[j];
			maxj = j;
		}
		TotalJ += (Expect[j] - TestY[j])*(Expect[j] - TestY[j])/2;
	}
	if(fabs(TestY[maxj] - 1) < 0.01)
		return 1;
	return 0;
}

void ReadW(ANN &T)
{
	inW = fopen("outW.txt", "r");
	for(int i = 0; i < Num_X; i ++)
		for(int j = 0; j < Num_Y; j ++)
			fscanf(inW, "%lf", &T.w[i][j]);
	fclose(inW);
}

void WriteW(ANN &T)
{
	outW = fopen("outW.txt", "w");
	for(int i = 0; i < Num_X; i ++)
		for(int j = 0; j < Num_Y; j ++)
			fprintf(outW, "%.15f ", T.w[i][j]);
	fclose(outW);
}