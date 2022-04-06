function loadJson(selector) {
    return JSON.parse(document.querySelector(selector).getAttribute('data-json'));
}


window.onload = function () {
    var jsonData = loadJson('#jsonData');
    var selected_company = loadJson('#selected_company');

    var data = jsonData.map((item) => item.data);
    var value = jsonData.map((item) => item.value);

    var config = {
        type: 'line',
        data: {
        labels: data,
        datasets: [
            {
              label: `График кампании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
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

    var ctx = document.getElementById('myChart').getContext('2d');
    window.myLine = new Chart(ctx, config);
};

