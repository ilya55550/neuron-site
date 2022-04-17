function loadJson(selector) {
    return JSON.parse(document.querySelector(selector).getAttribute('data-json'));
}


window.onload = function () {
    var data_for_graphic_with_predict = loadJson('#data_for_graphic_with_predict');
    var selected_company = loadJson('#selected_company');

//data_for_graphic_with_predict[0]  # дата дефолт графика
//data_for_graphic_with_predict[1]  # значения дефолт графика
//data_for_graphic_with_predict[2]  # дата прогноза
//data_for_graphic_with_predict[3]  # значения прогноза


    var config = {
        type: 'line',
        data: {
        labels: data_for_graphic_with_predict[0],
        datasets: [
            {
              label: `Фактические значения компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
//              backgroundColor: 'black',
              borderColor: 'green',
              pointRadius: 0,
              data: data_for_graphic_with_predict[1],
              fill: false
            },
            {
              label: `Прогноз компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
//              backgroundColor: 'black',
              borderColor: 'red',
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

//        var config2 = {
//        type: 'line',
//        data: {
//        labels: data_for_graphic_with_predict[2],
//        datasets: [
//            {
//              label: `Прогноз компании: ${selected_company[0]}, тикер: ${selected_company[1]}`,
////              backgroundColor: 'black',
//              borderColor: 'lightblue',
//              pointRadius: 0,
//              data: data_for_graphic_with_predict[3],
//              fill: false
//            }
//          ]
//        },
//        options: {
//        responsive: true
//        }
//    };


    var ctx = document.getElementById('myChart').getContext('2d');
//    var ctx2 = document.getElementById('myChart2').getContext('2d');

    window.myLine = new Chart(ctx, config);
//    window.myLine2 = new Chart(ctx2, config2);


};