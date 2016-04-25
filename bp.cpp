#include <stdio.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <time.h>
#include <math.h>

struct IOData {
	IOData(){
		this->x = this->y = 0;
	}
	IOData(int x, int y) {
		this->x = x;
		this->y = y;
	}
	int x, y;
};

IOData generateDataFunction(int x) {
	return IOData(x, 3 * x);
}

std::vector<IOData> generateData(int num) {
	std::vector<IOData> res(num);
	for (int i = 0; i < num; ++i) {
		res[i] = generateDataFunction(i);
	}
	return res;
}

struct NormalizationData {
	NormalizationData() {}
	NormalizationData(double d) {
		this->d = d;
	}
	double d;
};

template<class T>
T min(std::vector<T> container) {
	typename std::vector<T>::iterator it = container.begin();
	T res = 0x7fffffff;
	for (; it != container.end(); ++it) {
		if (res > *it) {
			res = *it;
		}
	}
	return res;
}

template<class T>
T max(std::vector<T> container) {
	typename std::vector<T>::iterator it = container.begin();
	T res = 0x80000000;
	for (; it != container.end(); ++it) {
		if (res < *it) {
			res = *it;
		}
	}
	return res;
}

double mapMinMax(int x, int min, int max) {
	return (double)(x-min) / (max - min);
}

template<class O, class I>
std::vector<O> map(std::vector<I> data, std::function<O(I)> callback) {
	std::vector<O> res(data.size());
	typename std::vector<I>::iterator it = data.begin();
	for(; it != data.end(); ++it) {
		res.push_back(callback(*it));
	}
	return res;
}

template<class T>
double sum(std::vector<T> data) {
	double res = 0;
	std::for_each(data.begin(), data.end(), [res](T d) { res += d;});
	return res;
}

std::vector<NormalizationData> normalization(std::vector<IOData> data) {
	std::vector<int> idata = map<int, IOData>(data, [](IOData data) {return data.x;});
	int ma = max(idata);
	int mi = min(idata);
	return map<NormalizationData, int>(idata, [ma, mi](int x) {return NormalizationData(mapMinMax(x, mi, ma));});
}

template<class T>
std::vector<T> rands(int num) {
	std::vector<T> res(num);
	for (int i = 0; i < num; ++i) {
		res.push_back(rand());
	}
	return res;
}

template<class T>
std::vector< std::vector<T> > rands(int num1, int num2) {
	std::vector< std::vector<T> > res(num1);
	for (int i = 0; i < num1; ++i) {
		for (int j = 0; j < num2; ++j) {
			res[i].push_back(rand());
		}
	}
	return res;
}

#define MATH_E (2.718281828459)
double logsig(double x) {
	return 1.0 / (1.0 + exp(-x));
}


double dot(std::vector<double> v1, std::vector<double> v2) {
	if (v1.size() != v2.size()) {
		return 0;
	}
	double res = 0;
	for (int i = 0; i < v1.size(); ++i) {
		res += v1[i] * v2[i];
	}
	return res;
}

int main(int argc, char *argv[])
{
	srand((unsigned)time(NULL));

	std::vector<IOData> trainData = generateData(10000);
	std::vector<NormalizationData> NormalizationData = normalization(trainData);

	int inputnum = 1;
	int outputnum = 1;
	int hidenum = 2;
	double ita = 0.5;

	std::vector<std::vector<int> > weightHide = rands<int>(hidenum, inputnum);
	std::vector<std::vector<int> > weightOutput = rands<int>(hidenum, outputnum);
	std::vector<int>  thresholdHide = rands<int>(hidenum);
	std::vector<int> thresholdOutput = rands<int>(outputnum);

	// train
	
	for (int oi  = 0;  oi< 20; oi++) {
		
		for (int i = 0; i < trainData.size(); ++i) {
			int x = trainData[i].x;
			int y = trainData[i].y;

			std::vector<double> H(hidenum);
			for (int j = 0; j < hidenum; ++j) {
				double tmp = x * weightHide[j][0]  + thresholdHide[j];
				double h = logsig(tmp);
				H.push_back(h);	

				double O = h * weightOutput[j][0] - thresholdHide[0];
				double e = y - O;

				weightHide[j][0] = weightHide[j][0] + ita * h * ( 1-h) * x * weightOutput[j][0] * e;
				weightOutput[j][0] = weightOutput[j][0] + ita * h * e;
				thresholdHide[j] = thresholdHide[j] + ita * h * (1-h) * weightOutput[j][0] * e;
				thresholdOutput[0] = thresholdOutput[0] + e;
			}

		}	
	}

	std::vector<int> idata = map<int, IOData>(trainData, [](IOData data) {return data.x;});
	int ma = max(idata);
	int mi = min(idata);
	for (int j = 0; j < hidenum; ++j) {
		double tmp = mapMinMax(50, mi, ma) * weightHide[j][0] + thresholdHide[j];
		double h = logsig(tmp);
		double O = weightOutput[j][0] * h + thresholdOutput[j];
		printf("%lf\n\n", O);
	}
		
	return 0;
}








