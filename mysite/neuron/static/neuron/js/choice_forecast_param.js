let csrftoken = getCookie('csrftoken');
let row = document.getElementsByClassName("row");
let row1 = document.getElementsByClassName("row1");
let item = document.getElementsByClassName("col-md-4 col-sm-4 col-xs-12 col-2");

function send_company(){
    let company_id = document.getElementById("id_company").value;
    console.log(company_id)
    post_data_company_id()

    async function post_data_company_id() {
        data = JSON.stringify({
            company_id: company_id,
        })

        const response = await fetch(async_handler, {
        method: 'POST',
        body: data,
        headers: { 'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json',
        "X-CSRFToken": csrftoken },
        })
        res = await response.json();

        while (item.length > 0) {
           item[0].remove();
        }

        console.log('start')
        let row1 = document.getElementsByClassName("row");
        let select_ip = document.createElement("div");
        select_ip.setAttribute("class", "col-md-4 col-sm-4 col-xs-12 col-2");

        row.parentNode.insertBefore(select_ip, row.nextSibling);

        console.log('finish')

    }
}


function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}