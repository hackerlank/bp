#include <stdio.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <time.h>
#include <math.h>

struct IData {
	IData() {}
	IData(std::vector<double> X) {
		this->X = X;
	}
	std::vector<double> X;
};

struct OData {
	std::vector<double> Y;
};

struct IOData {
	IOData(){}
	IOData(IData in, OData out) {
		this->input = in;
		this->output = out;
	}
	IData input;
	OData output;
};

IOData generateDataFunction(int x) {
	IData in;
	in.X.push_back(x);
	OData out;
	out.Y.push_back(3*x);
	return IOData(in, out);
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

	NormalizationData(std::vector<double> D) {
		this->D = D;
	}
	std::vector<double> D;
};

std::vector<double> min(std::vector<IData> container) {
	typename std::vector<double> res;
	for (int i = 0; i < container[0].X.size(); ++i) {
		double min = container[0].X[i];
		for (int j = 0; j < container.size(); ++j) {
			if (min > container[j].X[i]) {
				min = container[j].X[i];
			}	
		}
		res.push_back(min);
	}
	return res;
}

std::vector<double> max(std::vector<IData> container) {
	typename std::vector<double> res;
	for (int i = 0; i < container[0].X.size(); ++i) {
		double max = container[0].X[i];
		for (int j = 0; j < container.size(); ++j) {
			if (max < container[j].X[i]) {
				max = container[j].X[i];
			}	
		}
		res.push_back(max);
	}
	return res;
}

double mapMinMax(double x, double min, double max) {
	return (x-min) / (max - min);
}

template<class O, class I>
std::vector<O> map(std::vector<I> data, std::function<O(I)> callback) {
	std::vector<O> res;
	typename std::vector<I>::iterator it = data.begin();
	for(; it != data.end(); ++it) {
		res.push_back(callback(*it));
	}
	return res;
}

template<class O, class I>
std::vector<O> map(std::vector<I> data, std::function<O(I, int)> callback) {
	std::vector<O> res;
	typename std::vector<I>::iterator it = data.begin();
	int i = 0;
	for(; it != data.end(); ++it) {
		res.push_back(callback(*it, i));
		++i;
	}
	return res;
}

template<class T>
double sum(std::vector<T> data) {
	double res = 0;
	std::for_each(data.begin(), data.end(), [res](T d) { res += d;});
	return res;
}

//std::vector<NormalizationData> normalization(std::vector<IData> idata, std::vector<double> iMin, std::vector<double> iMax) {
	//std::vector<NormalizationData> res = map<NormalizationData, IData>(idata, [iMin, iMax](IData d) { 
			//return NormalizationData(map<double, double>(d.X, [iMin, iMax](double x, int i){ return mapMinMax(x, iMin[i], iMax[i]);})); 
		//});
	//return res;
//}

std::vector<NormalizationData> normalization(std::vector<IData> idata, std::vector<double> iMin, std::vector<double> iMax) {
	std::vector<NormalizationData> res;
	for (auto data : idata) {
		NormalizationData normalizationData;
		for (int i = 0; i < data.X.size(); ++i) {
			normalizationData.D.push_back(mapMinMax(data.X[i], iMin[i], iMax[i]));
		}
		res.push_back(normalizationData);
	}
	return res;
}

template<class T>
std::vector<T> rands(int num) {
	std::vector<T> res;
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

std::vector<double> vecCalc(std::vector<double> v1, std::vector<double> v2, std::function<double(double, double)> op) {
	if (v1.size() != v2.size()) {
		return std::vector<double>();	
	}

	return map<double, double>(v1, [op, v2](double a, int i) { return op(a, v2[i]);});
}

std::vector<double> mul(std::vector<double> v1, std::vector<double> v2) {
	return vecCalc(v1, v2, [](double a, double b) {return a * b;});
}

std::vector<double> mul(std::vector<double> v, double a) {
	return map<double, double>(v, [a](double b) {return a * b;});
}

std::vector<double> plus(std::vector<double> v1, std::vector<double> v2) {
	return vecCalc(v1, v2, [](double a, double b) {return a + b;});
}

std::vector<double> operator + (std::vector<double> v1, std::vector<double> v2) {
	return plus(v1, v2);
}

std::vector<double> operator * (std::vector<double> v1, std::vector<double> v2) {
	return mul(v1, v2);
}

std::vector<double> operator * (std::vector<double> v, double a) {
	return mul(v, a);
}

std::vector<double> operator * (double a, std::vector<double> v) {
	return mul(v, a);
}

void printf(std::vector<double> v) {
	std::for_each(v.begin(), v.end(), [](double d) {printf("%lf ", d);});
	printf("\n");
}

int main(int argc, char *argv[])
{
	srand((unsigned)time(NULL));

	std::vector<IOData> trainData = generateData(10000);
	std::vector<IData> idata = map<IData, IOData>(trainData, [](IOData data) {return data.input;});
	std::vector<OData> odata = map<OData, IOData>(trainData, [](IOData data) {return data.output;});
	std::vector<double> iMin = min(idata);
	std::vector<double> iMax = max(idata);
	std::vector<NormalizationData> normalizationData = normalization(idata, iMin, iMax);
	idata = map<IData, NormalizationData>(normalizationData, [](NormalizationData data) {return IData(data.D);});

	int inputnum = 1;
	int outputnum = 1;
	int hidenum = 2;
	double ita = 0.5;

	std::vector<std::vector<double> > weightHide = rands<double>(hidenum, inputnum);
	std::vector<std::vector<double> > weightOutput = rands<double>(outputnum, hidenum);
	std::vector<double>  thresholdHide = rands<double>(hidenum);
	std::vector<double> thresholdOutput = rands<double>(outputnum);

	// train
	
	for (int oi  = 0;  oi< 20; oi++) {

		for (int iditer = 0; iditer < idata.size(); ++iditer) {
			std::vector<double> H;
			for (int i = 0; i < hidenum; ++i) {
				double h = dot(idata[iditer].X, weightHide[i]);
				double h2 = logsig(h);
				H.push_back(h2);

			}	

			std::vector<double> O;
			for (int i = 0; i < outputnum; ++i) {
				double o = dot(H, weightOutput[i]);
				O.push_back(o);
			}


			std::vector<double> E = map<double, double>(odata[iditer].Y, [O](double y, int i) {return y - O[i];});
			for (int j = 0; j < hidenum; ++j) {
				for (int k = 0; k < outputnum; ++k) {
					weightOutput[k][j] = weightOutput[k][j] + ita * H[j] * E[k];
				}
			}
			//for (int i = 0; i < hidenum; ++i) {
				//weightOutput[i] = weightOutput[i] + ita * E * O;
			//}

			for (int i = 0; i < inputnum; ++i) {
				for (int j = 0; j < hidenum; ++j) {
					double sum = 0;
					for (int k = 0; k < outputnum; ++k) {
						sum += weightOutput[k][j] * E[k];	
					}
					weightHide[j][i] = weightHide[j][i] + ita * H[j] * (1 - H[j]) * idata[iditer].X[i] * sum;
				}
			}
		}

	}

	printf("weightHide:\n");
	printf(weightHide[0]);
	printf(weightHide[1]);
	printf("weightOutput:\n");
	printf(weightOutput[0]);

	std::vector<IOData> testData = generateData(2);
	std::vector<IData> itdata = map<IData, IOData>(testData, [](IOData data) {return data.input;});
	std::vector<OData> otdata = map<OData, IOData>(testData, [](IOData data) {return data.output;});
	//std::vector<double> itMin = min(itdata);
	//std::vector<double> itMax = max(itdata);
	std::vector<NormalizationData> tnormalizationData = normalization(itdata, iMin, iMax);
	itdata = map<IData, NormalizationData>(tnormalizationData, [](NormalizationData data) {return IData(data.D);});
		
	for (int iditer = 0; iditer < itdata.size(); ++iditer) {
		std::vector<double> H;
		for (int i = 0; i < hidenum; ++i) {
			double h = dot(itdata[iditer].X, weightHide[i]);
			double h2 = logsig(h);
			H.push_back(h2);

		}	

		std::vector<double> O;
		for (int i = 0; i < outputnum; ++i) {
			double o = dot(H, weightOutput[i]);
			O.push_back(o);
		}
		printf(itdata[iditer].X);
		printf(O);
	}
	
	return 0;
}








