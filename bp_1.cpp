#pragma once
#include<cassert>
#include<string>
using std::string;

enum Params
{
	InLayerNodesNum=1,//输入层节点数
	MidLayerNodesNum=8,//隐层节点数
	OutLayerNodesNum=1
};

/**
*  BP神经网络train的目的是找出这样的一组权值	_m_weight[Params::MidLayerNodesNum];//隐含层权值
*												_o_weight[Params::MidLayerNodesNum];//网络层权值
*  使输出与训练样本的真实输出间的误差小于阈值   _err
*  如果训练过程中迭代次数超出阈值 _it_nums 则认为训练失败 
*/

class BPNet
{
public:
	BPNet(){init();}
	~BPNet(){}
	
	int train(const double(*p)[Params::InLayerNodesNum],const double (*t)[Params::OutLayerNodesNum],int size);//
	int sim(const double(*p)[Params::InLayerNodesNum],double(*t)[Params::OutLayerNodesNum],int size);//
	bool save(){return save("data.b");}
	bool load(){return load("data.b");}
	bool save(const string& file);
	bool load(const string& file);

private:
	void init();
	void normalize(double **t,int size);//归一化处理
	void unnormalize(double **t,int size);//反归一化处理
private:
	
	double _m_w_rate;//输入层-隐含层权值学习率
	double _o_w_rate;//隐含层-网络层权学习率
	double _m_t_rate;//输入层-隐含层阈值学习率
	double _o_t_rate;//隐含层-网络层阈值学习率

	int _it_nums;//迭代次数
	double _err;//误差限

	double _mint,_maxt;
	
	double _m_threshold[Params::MidLayerNodesNum];//输入层-隐含层阈值
	double _o_threshold[Params::OutLayerNodesNum];//隐含层-网络层阈值
	double _m_weight[Params::InLayerNodesNum][Params::MidLayerNodesNum];//输入层-隐含层权值
	double _o_weight[Params::MidLayerNodesNum][Params::OutLayerNodesNum];//隐含层-网络层权值
};


#include<cstdlib>
#include<ctime>
#include<cmath>

#include<fstream>
#include<iostream>

using std::ifstream;
using std::ofstream;


int BPNet::train(const double(*p)[Params::InLayerNodesNum],const double (*t)[Params::OutLayerNodesNum],int size)
{//
	double o1[Params::MidLayerNodesNum];
	double o2[Params::OutLayerNodesNum];
	double error1[Params::MidLayerNodesNum];
	double error2[Params::OutLayerNodesNum];
	double max_error=0;//记录最大误差

	double *perr=new double[size];

	double** t_temp=new double*[size];//保存t的数据，这里的设计很糟糕
	for(int i=0;i<size;++i)
	{
		t_temp[i]=new double[Params::OutLayerNodesNum];
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			t_temp[i][j]=t[i][j];
	}

	normalize(t_temp,size);//归一化处理

	int i=0;
	for(;i<_it_nums;++i)
	{
		for(int j=0;j<size;++j)
		{
			//正向传播
				//计算隐含层输出
			for(int k=0;k<Params::MidLayerNodesNum;++k)
			{
				o1[k]=_m_weight[0][k]*p[j][0];
				for(int m=1;m<Params::InLayerNodesNum;++m)
					o1[k]+=_m_weight[m][k]*p[j][m];
				//激励函数
				o1[k]=1.0/(1+exp(-o1[k]-_m_threshold[k]));//隐含层各单元的输出
			}
				//计算输出层输出
			for(int k=0;k<Params::OutLayerNodesNum;++k)
			{
				o2[k]=_o_weight[0][k]*o1[0];
				for(int m=1;m<Params::MidLayerNodesNum;++m)
					o2[k]+=_o_weight[m][k]*o1[m];
				//激励函数
				o2[k]=1.0/(1+exp(-o2[k]-_o_threshold[k]));//隐含层各单元的输出 
			}

			//反向传播
			for(int k=0;k<Params::OutLayerNodesNum;++k)
			{
				//计算输出层误差
				error2[k]=(t_temp[j][k]-o2[k])*o2[k]*(1-o2[k]);
				//调整权值
				for(int m=0;m<Params::MidLayerNodesNum;++m)
					_o_weight[m][k]+=_o_w_rate*error2[k]*o1[m];
			}
				
			for(int k=0;k<Params::MidLayerNodesNum;++k)
			{
				//计算隐含层误差
				double d=0;
				for(int m=0;m<Params::OutLayerNodesNum;++m)
					d+=error2[m]*_o_weight[k][m];
				//调整权值
				error1[k]=d*o1[k]*(1-o1[k]);

				for(int m=0;m<Params::InLayerNodesNum;++m)
					_m_weight[m][k]+=_m_w_rate*error1[k]*p[j][m];
			}

			double e=0;
			for(int k=0;k<Params::OutLayerNodesNum;++k)
				e+=fabs(t_temp[j][k]-o2[k])*fabs(t_temp[j][k]-o2[k]);
			
			perr[j]=e/2;
			//更新阈值
			for(int k=0;k<OutLayerNodesNum;k++)  
				_o_threshold[k]+=_o_t_rate*error2[k]; //下一次的隐含层和输出层之间的新阈值  
			for(int k=0;k<MidLayerNodesNum;k++)  
				_m_threshold[k]+=_m_t_rate*error1[k]; //下一次的输入层和隐含层之间的新阈值  
		}

		max_error=perr[0];
		for(int j=1;j<size;++j)
			if(perr[j]>max_error)
				max_error=perr[j];
		if(max_error<_err)
			break;
	}

	delete[] perr;

	for(int i=0;i<size;++i)
		delete[] t_temp[i];
	delete[] t_temp;

	if(i>=_it_nums)
		return 0;
	return 1;
}

int BPNet::sim(const double(*p)[Params::InLayerNodesNum],double(*t)[Params::OutLayerNodesNum],int size)
{//
	double maxt,mint;

	double o1[Params::MidLayerNodesNum];

	double** t_temp=new double*[size];//保存t的数据，这里的设计很糟糕
	for(int i=0;i<size;++i)
		t_temp[i]=new double[Params::OutLayerNodesNum];

	for(int i=0;i<size;++i)
	{
		//正向传播
			//计算隐含层输出
		for(int k=0;k<Params::MidLayerNodesNum;++k)
		{
			o1[k]=_m_weight[0][k]*p[i][0];
			for(int m=1;m<Params::InLayerNodesNum;++m)
				o1[k]+=_m_weight[m][k]*p[i][m];
			o1[k]=1.0/(1.0+exp(-o1[k]-_m_threshold[k]));//隐含层各单元的输出 
		}
			//计算输出层输出
		for(int k=0;k<Params::OutLayerNodesNum;++k)
		{
			t_temp[i][k]=_o_weight[0][k]*o1[0];
			for(int m=1;m<Params::MidLayerNodesNum;++m)
				t_temp[i][k]+=_o_weight[m][k]*o1[m];
			t_temp[i][k]=1.0/(1.0+exp(-t_temp[i][k]-_o_threshold[k]));//隐含层各单元的输出 
		}
	}

	unnormalize(t_temp,size);//反归一化
	for(int i=0;i<size;++i)
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			t[i][j]=t_temp[i][j];

	for(int i=0;i<size;++i)
		delete[] t_temp[i];
	delete[] t_temp;

	return 1;
}

void BPNet::init()
{
	_m_w_rate=0.9;//输入层-隐含层权值学习率
	_o_w_rate=0.9;//隐含层-网络层权值学习率
	_m_t_rate=0.9;//输入层-隐含层权值学习率
	_o_t_rate=0.9;//隐含层-网络层权值学习率

	_it_nums=1000;//迭代次数
	_err=0.00001;//误差限
	
	//初始化成随机数
	srand(time(0));
	for(int i=0;i<Params::InLayerNodesNum;++i)
	{
		for(int j=0;j<Params::MidLayerNodesNum;++j)
			_m_weight[i][j]=(2.0*(double)rand()/RAND_MAX)-1;
	}

	for(int i=0;i<Params::MidLayerNodesNum;++i)
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			_o_weight[i][j]=(2.0*(double)rand()/RAND_MAX)-1;
	}
	for(int i=0;i<Params::MidLayerNodesNum;++i)
		_m_threshold[i]=(2.0*(double)rand()/RAND_MAX)-1;
	for(int i=0;i<Params::OutLayerNodesNum;++i)
		_o_threshold[i]=(2.0*(double)rand()/RAND_MAX)-1;
}

bool BPNet::save(const string& file)
{
	ofstream ofile(file);
	for(int i=0;i<Params::InLayerNodesNum;++i)//保存输入层-隐含层权值
	{
		for(int j=0;j<Params::MidLayerNodesNum;++j)
			ofile<<_m_weight[i][j]<<' ';
		ofile<<'\n';
	}

	for(int i=0;i<Params::MidLayerNodesNum;++i)//保存隐含层-网络层权值
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			ofile<<_o_weight[i][j]<<' ';
		ofile<<'\n';
	}
	for(int i=0;i<Params::MidLayerNodesNum;++i)//保存输入层-隐含层阈值
		ofile<<_m_threshold[i]<<" ";
	ofile<<'\n';

	for(int i=0;i<Params::OutLayerNodesNum;++i)//保存隐含层-网络层阈值
		ofile<<_o_threshold[i]<<" ";
	ofile.close();
	return true;
}

bool BPNet::load(const string& file)
{
	ifstream ifile(file);
	char endLine;
	for(int i=0;i<Params::InLayerNodesNum;++i)//加载输入层-隐含层权值
	{
		for(int j=0;j<Params::MidLayerNodesNum;++j)
			ifile>>_m_weight[i][j];
	}

	for(int i=0;i<Params::MidLayerNodesNum;++i)//加载隐含层-网络层权值
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			ifile>>_o_weight[i][j];
	}
	for(int i=0;i<Params::MidLayerNodesNum;++i)//加载输入层-隐含层阈值
		ifile>>_m_threshold[i];

	for(int i=0;i<Params::OutLayerNodesNum;++i)//加载隐含层-网络层阈值
		ifile>>_o_threshold[i];

	ifile.close();
	return true;
}

void BPNet::normalize(double **t,int size)
{//归一化处理
	_mint=0x7FFFFFFF;//这里利用了“魔数”进行处理，不是最好的方法
	_maxt=-0x7FFFFFFF;

	//找到最大和最小值
	for(int i=0;i<size;++i)
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
		{
			if(_mint>t[i][j])
				_mint=t[i][j];
			if(_maxt<t[i][j])
				_maxt=t[i][j];
		}
	}

	if(_mint==_maxt)
		_mint=0;

	double range=_maxt-_mint;

	//归一化
	//找到最大和最小值
	for(int i=0;i<size;++i)
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			t[i][j]=(t[i][j]-_mint)/range;
	}
}

void BPNet::unnormalize(double **t,int size)
{//反归一化处理
	double range=_maxt-_mint;

	//归一化
	//找到最大和最小值
	for(int i=0;i<size;++i)
	{
		for(int j=0;j<Params::OutLayerNodesNum;++j)
			t[i][j]=t[i][j]*range+_mint;
	}
}






#include<cstdlib>
#include<ctime>
#include<iostream>
using std::cout;
using std::cin;
using std::endl;




int main()
{
	BPNet bpNet;
	const int N=500; 
	const double Pi=3.1415926;

	srand(time(0));

	double a[N][Params::InLayerNodesNum];
	double a2[N][Params::InLayerNodesNum];
	double b[N][Params::OutLayerNodesNum];
	double b2[N][Params::OutLayerNodesNum];
	double b3[N][Params::OutLayerNodesNum];

	for(int i=0;i<N;++i)
	{
		//a[i][0]=((2.0*(double)rand()/RAND_MAX)-1)*(Pi/2-1)+1;
		//a2[i][0]=((2.0*(double)rand()/RAND_MAX)-1)*(Pi/2-1)+1;

		//b[i][0]=2*sin(a[i][0])-0.7;
		//b2[i][0]=2*sin(a2[i][0])-0.7;
		
		a[i][0]= rand() % 1000;
		a2[i][0]= rand() % 1000;

		b[i][0]= a[i][0] * a[i][0];
		b2[i][0]= a2[i][0]* a2[i][0];
	}

	bpNet.train(a,b,N);
	bpNet.save();
	bpNet.load();
	bpNet.sim(a,b3,N);

	for(int i=0;i<N;++i)
		cout<<a[i][0]<<" -> "<<b3[i][0]<<endl;

	return 0;
}


