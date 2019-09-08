/*
def rbf(X,Y, gamma):    
    # X,Y is a Python/JavaScript list instead of numpy.array
    
    norm = 0.0
    for i in range(len(X)):
        norm += (X[i] - Y[i])**2
    
    return math.exp(-gamma * norm)

*/

function rbf(X, Y, gamma){
    var norm = 0.0;
    var i;
    for(i=0;i<X.length;i++){
        norm += Math.pow(X[i] - Y[i],2);
    }
    
    return Math.exp(-gamma * norm)
}

/*
def predict(X, support_vectors, dual_coef, intercept, gamma):
    pred = intercept
    for i in range(len(support_vectors)):
        w = rbf(X, support_vectors[i], gamma)
        pred += w * dual_coef[i]
    return pred
*/

function predict(X, support_vectors, dual_coef, intercept, gamma){
    var pred = intercept;
    var i;
    for(i=0;i<support_vectors.length;i++){
        var w = rbf(X, support_vectors[i], gamma);
        pred += w * dual_coef[i];
    }
    return pred
}

var UI = {};
var idList = ['age', 'Scr', 'Cys', 'sex', 'rGFR', 'compute'];
var i=0;
for(i=0;i<idList.length;i++){
    var id = idList[i];
    var el = document.getElementById(id);
    UI[id] = el;
}

/*
inp_scaled = {}
for key in inp:
    if scale_map[key] == 'log':
        x = math.log(inp[key])
    else:
        x = inp[key]
    
    x_scaled = (x - mean_map[key])/std_map[key]
    inp_scaled[key] = x_scaled
*/

function compute(inp){
    var inp_scaled = {};
    Object.keys(inp).forEach(function(key){
        var x;
        if(frozen.scale_map[key] == 'log'){
            x = Math.log(inp[key]);
        }else{
            x = inp[key];
        }
        x_scaled = (x - frozen.mean_map[key])/frozen.std_map[key]
        inp_scaled[key] = x_scaled;
    })

    var inp_vec = [];
    frozen.key_list.forEach(function(key){
        inp_vec.push(inp_scaled[key]);
    })

    var y_raw = predict(inp_vec, frozen.support_vectors, 
        frozen.dual_coef, frozen.intercept, frozen.gamma);
    
    var y;
    if(frozen.y_log){
        y = Math.exp(y_raw);
    }else{
        y = r_raw
    }

    return y;
}

function run(){
    var age = Number(UI['age'].value);
    var Scr = Number(UI['Scr'].value);
    var Cys = Number(UI['Cys'].value);
    var sex = Number(UI['sex'].value);

    inp = {age: age, Scr:Scr, Cys:Cys, sex:sex};

    var y = compute(inp);

    console.log(inp,"->",y);

    UI['rGFR'].value = y;
}
/*
inp_vec = []
for key in key_list:
    inp_vec.append(inp_scaled[key])

y_raw = predict(inp_vec, support_vectors, dual_coef, intercept, gamma)

if y_log:
    y = math.exp(y_raw)
else:
    y = y_raw
*/

UI['compute'].onclick = run;
