const tf = require('@tensorflow/tfjs');

class LinearRegression {
constructor(features,labels,options){
   this.features = this.processFeatures(features);
    this.labels= tf.tensor(labels);
    this.mseHistroy=[];
    this.options= Object.assign({ learningRate:0.1,iteration:1000},options);
    this.weights =tf.zeros([this.features.shape[1],1]);
    
    
}
gradientDescent(){
    const currentGusses=this.features.matMul(this.weights);
    const difference =currentGusses.sub(this.labels);
    const slopes= this.features
    .transpose()
    .matMul(difference)
    .div(this.features.shape[0])
    this.weights=this.weights.sub(slopes.mul(this.options.learningRate))
}
train(){
    
    for(let i=0;i<this.options.iteration;i++){
        // console.log(this.options.learningRate);
       
        this.gradientDescent();
        this.mseRecord();
         this.updateLearningRate();
    }
}
   
    test(testFeatures,testLabels){
        testFeatures= this.processFeatures(testFeatures)
        testLabels=tf.tensor(testLabels); 
       
        const predication= testFeatures.matMul(this.weights);
        const res = testLabels.sub(predication)
        .pow(2)
        .sum()
        .get();
        const tot = testLabels.sub(testLabels.mean())
        .pow(2)
        .sum()
        .get();
        return 1-res/tot;
    }
    predict(observations){
        return this.processFeatures(observations).matMul(this.weights);
        
}
processFeatures(features){
    features =tf.tensor(features);
   
    if(this.mean && this.variance){
        features = features.sub(this.mean).div(this.variance.pow(0.5))
    }
    else{
        features= this.standerize(features);
    }
    features = tf.ones([features.shape[0],1]).concat(features,1);
    return features
}
 standerize(features){
     const{mean,variance}= tf.moments(features,0);
     this.mean = mean;
     this.variance =variance;
     return features.sub(this.mean).div(this.variance.pow(0.5));
 }
 mseRecord(){
    const mse= this.features
     .matMul(this.weights)
     .sub(this.labels)
     .pow(2)
     .sum()
     .div(this.features.shape[0])
     .get()
  this.mseHistroy.unshift(mse)
 }
 updateLearningRate(){
     if(this.mseHistroy.length<2)
     {
         return ;
     }
    if(this.mseHistroy[0]>this.mseHistroy[1]){
    this.options.learningRate/=2;
    }
    else{
        this.options.learningRate*=1.05;
    }
 }
}
module.exports = LinearRegression;