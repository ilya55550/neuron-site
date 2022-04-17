function loadJson(selector) {
    return JSON.parse(document.querySelector(selector).getAttribute('data-json'));
}


window.onload = function () {
    var data_for_graphic_with_predict = loadJson('#data_for_graphic_with_predict');
//    var res_date = loadJson('#res_date');
//    var res_volume = loadJson('#res_volume');
//    var predict_date = loadJson('#predict_date');
//    var predict_value = loadJson('#predict_value');


    var selected_company = loadJson('#selected_company');
    var data_for_graphic = loadJson('#jsonData');

//data_for_graphic_with_predict[0]  # дата дефолт графика
//data_for_graphic_with_predict[1]  # значения дефолт графика
//data_for_graphic_with_predict[2]  # дата прогноза
//data_for_graphic_with_predict[3]  # значения прогноза


    var data = data_for_graphic.map((item) => item.data);
    var value = data_for_graphic.map((item) => item.value);

    var config = {
        type: 'line',
        data: {
        labels: data,
        datasets: [
            {
              label: `График компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
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
        labels: data_for_graphic_with_predict[0],
        datasets: [
            {
              label: `График компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
//              backgroundColor: 'black',
              borderColor: 'lightblue',
              pointRadius: 0,
              data: data_for_graphic_with_predict[1],
              fill: false
            }
          ]
        },
        options: {
        responsive: true
        }
    };

            var config3 = {
        type: 'line',
        data: {
        labels: data_for_graphic_with_predict[2],
        datasets: [
            {
              label: `Прогноз компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
//              backgroundColor: 'black',
              borderColor: 'lightblue',
              pointRadius: 0,
              data: data_for_graphic_with_predict[3],
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
    var ctx3 = document.getElementById('myChart3').getContext('2d');

    window.myLine = new Chart(ctx, config);
    window.myLine2 = new Chart(ctx2, config2);
    window.myLine3 = new Chart(ctx3, config3);


};

