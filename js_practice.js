function solveMeFirst(a, b) {
    return a + b
}

function simpleArraySum(ar) {
    return ar.reduce((a, c) => a + c)
}

function compareTriplets(a, b) {
    let score = [0,0]
    for (let i = 0; i < a.length; i ++){
        if(a[i] === b[i]){
            continue
        } else if (a[i] > b[i]){
            score[0]++
        } else {
            score[1]++
        }
    }
    return score
}

function aVeryBigSum(ar) {
    return ar.reduce((a, c) => a + c)
}

function plusMinus(arr) {
    let res = [0,0,0]
    arr.forEach((ele) => {
        if(ele === 0){
            res[2]++
        } else if (ele > 0){
            res[0]++
        } else{
            res[1]++
        }
    })
    for(let i = 0; i < res.length; i++){
        console.log(+(res[i] / arr.length).toFixed(6))
    }
}

function staircase(n) {
    let temp = 1
    while(n > 0){
        console.log(" ".repeat(n - 1) + "#".repeat(temp))
        temp ++ 
        n --
    }
}