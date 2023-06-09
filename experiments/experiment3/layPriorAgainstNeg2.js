
// we want to include a mechanism where inferring one emotion explains away another emotion
var emoGen_mcmc_samples = 1e3
var emoGen_mcmc_burn = 1e2
var layTheory_nData = 1e2
var layTheory_mcmc_samples = 1e3
var layTheory_mcmc_burn = 1e2
var joinModel_mcmc_samples = 2e3
var joinModel_mcmc_burn = 1e2

// could be a project about knowing your appraisal or somatic from your emotion
// could have an action output and be about knowing your emotion from the situation vs your action
// an emotionally aware person knows their appraisal better and their somatic better
// people might be biased away from feeling certain emotions


var emoGen = function(situation){
//   return Infer({method: 'MCMC', samples: emoGen_mcmc_samples, burn: emoGen_mcmc_burn, model: function(){
  return Infer({method: 'rejection', model: function(){
     
    // Define situation prior for 4 types of situations (i.e., core relational themes)
//     var situation = categorical({vs: ['offense', 'loss', 'accomplishment']})


    var congruence = (situation == 'loss' || situation == 'offense') ? flip(.05) :
    (situation == 'accomplishment') ? flip(.95) : 
    flip()
    
    var arousal = (situation == 'offense') ? gaussian({mu: 7, sigma: .5}) :
    (situation == 'loss') ? gaussian({mu: 3, sigma: .5}) :
    (situation == 'accomplishment') ? uniform(0, 10) : 
    gaussian({mu: 3, sigma: .5})
    
    // if a situation is congruent, happiness is likely; if incongruent, anger, sadness are likely
    var congruence2emoDist = congruence ?
    {
      anger: Beta({a: 1, b: 8}), 
      sadness: Beta({a: 1, b: 8}), 
      happiness: Beta({a: 8, b: 1})
    } : {
      anger: Beta({a: 8, b: 1}), 
      sadness: Beta({a: 8, b: 1}),
      happiness: Beta({a: 1, b: 8})
    }

    // if arousal is high, anger is likely; if low, sadness is likely
    var arousal2emoDist = arousal > 5 ?
    {
      anger: Beta({a: 8, b: 1}), 
      sadness: Beta({a: 1, b: 8}), 
      happiness: Beta({a: 1, b: 1})
    } : {
      anger: Beta({a: 1, b: 8}),
      sadness: Beta({a: 8, b: 1}), 
      happiness: Beta({a: 1, b: 1})
    }
    
    // combining appraisal dimensions and arousal to infer emotion based on the situation
    // first, create a categorical distribution by sampling from each appraisal dimension and arousal, as defined above; then uniform sample from that distribution (default of 'categorical')
    var inferredAngerDist = Mixture({dists: 
      [congruence2emoDist['anger'], 
      arousal2emoDist['anger']],
      ps: [1, 1]})

    var inferredSadnessDist = Mixture({dists:
      [congruence2emoDist['sadness'], 
      arousal2emoDist['sadness']],
      ps: [1, 1]})

    var inferredHappinessDist = Mixture({dists:
      [congruence2emoDist['happiness'], 
      arousal2emoDist['happiness']],
      ps: [1, 1]})


    var anger = sample(inferredAngerDist)
    var sadness = sample(inferredSadnessDist)
    var happiness = sample(inferredHappinessDist)

    // condition(anger < .5)
    // condition(sadness < .5)

//     condition(situation == sit)
    return {
      anger: anger, 
      sadness:  sadness,
      happiness: happiness
    }
    
  }})
}

// var x = emoGen('transgression')
// viz.marginals(x)
// print(expectation(marginalize(x, 'anger')))
// print(expectation(marginalize(x, 'sadness')))
// print(expectation(marginalize(x, 'happiness')))
/***** Generate observed data ******/
var makeData = function(){
  var sit = categorical({vs: ['loss', 'accomplishment', 'offense']})
  var inferredEmos = sample(emoGen(sit))
  var anger = inferredEmos['anger']
  var happiness = inferredEmos['happiness']
  var sadness = inferredEmos['sadness']
  return {situation: sit, 
    anger: anger, 
    happiness: happiness, 
    sadness: sadness}
}
var observedData = repeat(layTheory_nData, makeData)
// viz(observedData)
// print(observedData)
// print(Math.round(observedData[0]['anger']))

/***** Lay theory ******/
var layTheory = function(sit){
  return Infer({method: 'MCMC', samples: layTheory_mcmc_samples, burn: layTheory_mcmc_burn}, function(){

    var negAversion = 1
    
    var angerProb = mem(function(state) { return beta(1, negAversion) })
    var happinessProb = mem(function(state) { return beta(1, 1) })
    var sadnessProb = mem(function(state) { return beta(1, negAversion) })

    var obsFn = function(datum){
      observe(Bernoulli({p: angerProb(datum['situation'])}), Math.round(datum['anger'])==1)
      observe(Bernoulli({p: happinessProb(datum['situation'])}), Math.round(datum['happiness'])==1)
      observe(Bernoulli({p: sadnessProb(datum['situation'])}), Math.round(datum['sadness'])==1)
    }
    mapData({data: observedData}, obsFn)
    
    return  {
      anger: angerProb(sit), 
      sadness: sadnessProb(sit), 
      happiness: happinessProb(sit)
    }
})}
// var x = layTheory('loss')
// // var x = layTheory('offense')
// // var x = layTheory('accomplishment')
// print(expectation(marginalize(x, 'anger')))
// print(expectation(marginalize(x, 'sadness')))
// print(expectation(marginalize(x, 'happiness')))
// viz.marginals(x)


/***** Integration model ******/
var joiningModel = function(sit){
  return Infer({method: 'MCMC', samples: joinModel_mcmc_samples, burn: joinModel_mcmc_burn}, function(){
    var egDist = emoGen(sit)
    var ltDist = layTheory(sit)
    var ltDistWeight = beta(1, 1)
//     var ltDistWeight = .5
    var egDistWeight = 1 - ltDistWeight


    var anger = sample(Mixture({dists: [marginalize(egDist, 'anger'), 
     marginalize(ltDist, 'anger')], 
      ps: [egDistWeight, ltDistWeight]}))
    var sadness = sample(Mixture({dists: [marginalize(egDist, 'sadness'), 
      marginalize(ltDist, 'sadness')], 
      ps: [egDistWeight, ltDistWeight]}))
    var happiness = sample(Mixture({dists: [marginalize(egDist, 'happiness'), 
      marginalize(ltDist, 'happiness')], 
      ps: [egDistWeight, ltDistWeight]}))


    return {
      layTheoryWeight: ltDistWeight,
      anger: anger,
      sadness: sadness,
      happiness: happiness
    }
  })
}
var x = joiningModel('loss')
csv.writeJoint(x, 'distributions/layPriorAgainstNeg_baseline.csv')
// var x = joiningModel('offense')
// var x = joiningModel('accomplishment')
// print('layTheoryWeight: ' + expectation(marginalize(x, 'layTheoryWeight')))
// print('anger: ' + expectation(marginalize(x, 'anger')))
// print('sadness: ' + expectation(marginalize(x, 'sadness')))
// print('happiness: ' + expectation(marginalize(x, 'happiness')))
// viz.marginals(x)
