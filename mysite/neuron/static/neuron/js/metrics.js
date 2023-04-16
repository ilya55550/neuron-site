function loadJson(selector) {
    return JSON.parse(document.querySelector(selector).getAttribute('data-json'));
}


window.onload = function () {
    var accuracy = loadJson('#accuracy');
    var loss = loadJson('#loss');

    console.log(accuracy)
    console.log(loss)


    var epoch = accuracy.map((item) => item.epoch);
    var value = accuracy.map((item) => item.value);

    var epoch2 = loss.map((item) => item.epoch);
    var value2 = loss.map((item) => item.value);

    var config = {
        type: 'line',
        data: {
        labels: epoch,
        datasets: [
            {
              label: 'accuracy',
//              backgroundColor: 'black',
              borderColor: 'lightblue',
              pointRadius: 0,
              data: value,
              fill: false
            }
          ]
        },
        options: {
        responsive: true
        }
    };

        var config2 = {
        type: 'line',
        data: {
        labels: epoch2,
        datasets: [
            {
              label: 'loss',
//              backgroundColor: 'black',
              borderColor: 'lightblue',
              pointRadius: 0,
              data: value2,
              fill: false
            }
          ]
        },
        options: {
        responsive: true
        }
    };


    var ctx = document.getElementById('myChart').getContext('2d');
    var ctx2 = document.getElementById('myChart2').getContext('2d');

    window.myLine = new Chart(ctx, config);
    window.myLine2 = new Chart(ctx2, config2);


};

