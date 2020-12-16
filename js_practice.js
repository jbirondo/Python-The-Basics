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

function miniMaxSum(arr) {
    let sort = arr.sort((a, b) => a - b)
    let min = sort.slice(0, 4).reduce((a, c) => a + c)
    let max = sort.slice(1).reduce((a, c) => a + c)
    console.log(`${min} ${max}`)
}

function birthdayCakeCandles(candles) {
    let h = {}
    candles.forEach(ele => {
        if(h.hasOwnProperty(ele)){
            h[ele] = h[ele] + 1
        } else {
            h[ele] = 1
        }
    })
    return h[candles.sort((a, b) => a - b)[candles.length - 1]]
}

function timeConversion(s) {
    let arr = s.split(":")
    let h, m, sec
    [h, m, sec] = [arr[0], arr[1], arr[2]]
    if(sec.includes("P") && parseInt(h) !== 12){
        h = parseInt(arr[0]) + 12
    }
    if(sec.includes("A") && parseInt(h) === 12){
        h = "00"
    }
    return `${h}:${m}:${sec.slice(0,2)}`
}

function countApplesAndOranges(s, t, a, b, apples, oranges) {
    let [aIn, oIn] = [0,0]
    apples.forEach(ele => {
        if(a + ele >= s && a + ele <= t){
            aIn = aIn + 1
        }
    })
    oranges.forEach(ele => {
        if(b + ele >= s && b + ele <= t){
            oIn = oIn + 1
        }
    })
    console.log(aIn)
    console.log(oIn)
}

function kangaroo(x1, v1, x2, v2) {
    let far = x1 < x2 ? x1 : x2
    let near = x1 < x2 ? x2 : x1
    if(v1 <= v2 && far !== near){
        return "NO"
    }
    while (far <= near){
        if(far === near){
            return "YES"
        }
        far = far + v1
        near = near + v2
    }
    return "NO"
}

function gradingStudents(grades) {
    for(let i = 0; i < grades.length; i++){
        if((grades[i] % 5 === 4 || grades[i] % 5 === 3) && grades[i] >= 38){
            grades[i] = Math.round(grades[i]/5) * 5
        }
    }
    return grades
}

function breakingRecords(scores) {
    let [min, max] = [scores[0], scores[0]]
    let [minC, maxC] = [0, 0]
    scores.forEach(score => {
        if(score > max){
            maxC++
            max = score
        } else if (score < min){
            minC++
            min = score
        }
    })
    return [maxC, minC]
}

function birthday(s, d, m) {
    let counter = 0
    for(let i = 0; i + m < s.length + 1; i++){
        s.slice(i, i + m).reduce((a,c) => a + c) === d ? counter++ : counter = counter
    }
    return counter
}

function divisibleSumPairs(n, k, ar) {
    let counter = 0
    for(let i = 0; i < ar.length; i++){
        for(let j = i + 1; j < ar.length; j++){
            if((ar[i] + ar[j]) % k === 0) counter++
        }
    }
    return counter
}

function migratoryBirds(arr) {
    let h = {}
    arr.forEach(bird => h.hasOwnProperty(bird) ? h[bird]++ :h[bird] = 1)
    let list = Object.entries(h)
        .sort(([,a],[,b]) => b-a)
    return list[0][0]
    // var birdsObj = {};

    // for (let i = 0; i < types.length; i++){
    //     let birdType = types[i];
    //     birdsObj[birdType] ? birdsObj[birdType]++ : birdsObj[birdType] = 1;
    // }

    // var mostBirds = 0;
    // var mostBirdType;
    // for (let birdType in birdsObj){
    //     if (birdsObj[birdType] > mostBirds){
    //         mostBirds = birdsObj[birdType];
    //         mostBirdType = birdType;
    //     }
    // }

    // console.log(mostBirdType);
}

function bonAppetit(bill, k, b) {
    bill.splice(k, 1)
    let total = bill.reduce((a, c) => a + c)/2 
    let res = total === b ? "Bon Appetit" : b - total
    console.log(res)
}

function climbingLeaderboard(scores, alice) {
    let res = []
    alice.forEach(score => {
        scores.push(score)
        let rank = [...new Set(scores.sort((a, b) => b - a))]
        res.push(rank.indexOf(score) + 1)
    })
    return res
    // var rscores = scores.reduce((score, currentScore, i) => {
    //     if(score[score.length-1] !== currentScore){
    //         score.push(currentScore);
    //         return score;
    //     } else {
    //         return score;
    //     }
    // }, []);
    // var initRank = rscores.length;
    // for(var i of alice){
    //     while(rscores[initRank-1] <= i){
    //         initRank--;
    //     }
    //     console.log(initRank+1);
    // }
}

function extraLongFactorials(n) {
    let res = 1
    while(n > 1) {
        res = BigInt(res) * BigInt(n)
        n--
    }
    console.log(res.toString())
}

function appendAndDelete(s, t, k) {
    while(s[0] === t[0] && s.length > 0 && t.length > 0){
        s = s.slice(1)
        t = t.slice(1)
    }
    let res = (s.length + t.length) <= k  ? "Yes": "No"
    return res
}