require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
let loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regressions')
let{features,labels,testFeatures,testLabels}=loadCSV('./cars.csv',{
    shuffle:true,
    splitTest:50,
    dataColumns:['horsepower','displacement','weight'],
    labelColumns:['mpg']
});
const regression = new LinearRegression(features,labels,{
    learningRate:1,
    iterations:3,
    batchSize:10
})
regression.train();
regression.features.print();
console.log("m",regression.weights.get(1,0),'b',regression.weights.get(0,0));
const r2 = regression.test(testFeatures,testLabels);
console.log('accurecy',r2)
const prediction=regression.predict([
    [-42,-20.8,168.48]
])
 
 prediction.print()


