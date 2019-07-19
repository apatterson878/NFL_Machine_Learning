let teams = [{"abbr":"atl", "logo":"../static/logos/atl.gif", "name":"Atlanta Falcons"},
  {"abbr":"buf", "logo":"../static/logos/buf.gif", "name":"Buffalo Bills"}, 
  {"abbr":"car", "logo":"../static/logos/car.gif", "name":"Carolina Panthers"}, 
  {"abbr":"chi", "logo":"../static/logos/chi.gif", "name":"Chicago Bears"}, 
  {"abbr":"cin", "logo":"../static/logos/cin.gif", "name":"Cincinnati Bengals"}, 
  {"abbr":"cle", "logo":"../static/logos/cle.gif", "name":"Cleveland Browns"}, 
  {"abbr":"clt", "logo":"../static/logos/clt.gif", "name":"Indianapolis Colts"}, 
  {"abbr":"crd", "logo":"../static/logos/crd.gif", "name":"Arizona Cardinals"}, 
  {"abbr":"dal", "logo":"../static/logos/dal.gif", "name":"Dallas Cowboys"}, 
  {"abbr":"den", "logo":"../static/logos/den.gif", "name":"Denver Broncos"}, 
  {"abbr":"det", "logo":"../static/logos/det.gif", "name":"Detroit Lions"}, 
  {"abbr":"gnb", "logo":"../static/logos/gnb.gif", "name":"Green Bay Packers"}, 
  {"abbr":"htx", "logo":"../static/logos/htx.gif", "name":"Houston Texans"}, 
  {"abbr":"jax", "logo":"../static/logos/jax.gif", "name":"Jacksonville Jaguars"}, 
  {"abbr":"kan", "logo":"../static/logos/kan.gif", "name":"Kansas City Chiefs"}, 
  {"abbr":"mia", "logo":"../static/logos/mia.gif", "name":"Miami Dolphins"}, 
  {"abbr":"min", "logo":"../static/logos/min.gif", "name":"Minnesota Vikings"}, 
  {"abbr":"nor", "logo":"../static/logos/nor.gif", "name":"New Orleans Saints"}, 
  {"abbr":"nwe", "logo":"../static/logos/nwe.gif", "name":"New England Patriots"}, 
  {"abbr":"nyg", "logo":"../static/logos/nyg.gif", "name":"New York Giants"}, 
  {"abbr":"nyj", "logo":"../static/logos/nyj.gif", "name":"New York Jets"}, 
  {"abbr":"oti", "logo":"../static/logos/oti.gif", "name":"Tennessee Titans"}, 
  {"abbr":"phi", "logo":"../static/logos/phi.gif", "name":"Philadelphia Eagles"}, 
  {"abbr":"pit", "logo":"../static/logos/pit.gif", "name":"Pittsburgh Steelers"}, 
  {"abbr":"rai", "logo":"../static/logos/rai.gif", "name":"Oakland Raiders"}, 
  {"abbr":"ram", "logo":"../static/logos/ram.gif", "name":"Los Angeles Rams"}, 
  {"abbr":"rav", "logo":"../static/logos/rav.gif", "name":"Baltimore Ravens"}, 
  {"abbr":"sdg", "logo":"../static/logos/sdg.gif", "name":"Los Angeles Chargers"}, 
  {"abbr":"sea", "logo":"../static/logos/sea.gif", "name":"Seattle Seahawks"}, 
  {"abbr":"sfo", "logo":"../static/logos/sfo.gif", "name":"San Francisco 49ers"}, 
  {"abbr":"tam", "logo":"../static/logos/tam.gif", "name":"Tampa Bay Buccaneers"}, 
  {"abbr":"was", "logo":"../static/logos/was.gif", "name":"Washington Redskins"}]


function buildCurrentSeasonPrediction(ml_type) {
    let url = `/prediction/current/${ml_type}`;
    d3.json(url).then(function (response) {
      let tableSelector = d3.select("#predictor-table");
      const predictionJson = response
      //Seasons = 16 Playoff Seasons.
      //Create the static Table Headers
      //https://gist.github.com/gka/17ee676dc59aa752b4e6
      //http://bl.ocks.org/AMDS/4a61497182b8fcb05906
      //https://vis4.net/blog/2015/04/making-html-tables-in-d3-doesnt-need-to-be-a-pain/
      let columns = [
        { head: 'Team', cl: 'table-head'},
        { head: 'Week 1', cl: 'table-head'},
        { head: 'Week 2', cl: 'table-head'},
        { head: 'Week 3', cl: 'table-head'},
        { head: 'Week 4', cl: 'table-head'},
        { head: 'Week 5', cl: 'table-head'},
        { head: 'Week 6', cl: 'table-head'},
        { head: 'Week 7', cl: 'table-head'},
        { head: 'Week 8', cl: 'table-head'},
        { head: 'Week 9', cl: 'table-head'},
        { head: 'Week 10', cl: 'table-head'},
        { head: 'Week 11', cl: 'table-head'},
        { head: 'Week 12', cl: 'table-head'},
        { head: 'Week 13', cl: 'table-head'},
        { head: 'Week 14', cl: 'table-head'},
        { head: 'Week 15', cl: 'table-head'},
        { head: 'Week 16', cl: 'table-head'},
        { head: 'Week 17', cl: 'table-head'}
      ];

      tableSelector.html("")
      let table = tableSelector.append("table")
      table.attr('class', "table table-striped table-bordered table-condensed").attr("id", "predictor-table-id")
      table.append('thead').append('tr')
      .selectAll('th')
      .data(columns).enter()
      .append('th')
      .attr('class', function (d) {
        return d.cl;
      })
      .text(function (d) {
        return d.head;
      });

      let tbody = table.append("tbody")
      //Iterate JSON
      predictionJson.forEach(mt =>{
        const row = tbody.append('tr')
        row.append("td").html( getTeamLogo(mt.MainTeam) + " " + getTeamName(mt.MainTeam) )
        const seasonsJ = mt.Seasons
        seasonsJ.forEach(seasonJ => {
          if(seasonJ.WinOrLoss == -1) {
            row.append("td").html("")
          } else {
            if(seasonJ.WinOrLoss == 1) {
              
              row.append("td").html(getTeamLogo(seasonJ.PlayedAgainst) + " " + "<a href=\"#\" class=\"badge badge-pill badge-success\"><i class=\"fas fa-trophy\"></i></a>")
            } else {
              row.append("td").html(getTeamLogo(seasonJ.PlayedAgainst) + " " + "<a href=\"#\" class=\"badge badge-light\"><i class=\"fas fa-times\"></i></a>")
            }
          }
        });//End Season Loop
      });//End Main Team Loop
      //Conclude the table

    });
}

function getTeamLogo(team) {
  let response = "<img src=\"";
  teams.forEach(t => {
    if(t.abbr === team) {
      response = response + t.logo + "\" alt = \""+t.name+"\" style=\"width:20px;height:20px;\">" ;
    }
  });
  return response
}

function getTeamName(team) {
  let response = "";
  teams.forEach(t => {
    if(t.abbr === team) {
      response = t.name;
    }
  });
  return response
}

function buildPrediction(season, ml_type) {
  let url = `/prediction/${season}/${ml_type}`;
  d3.json(url).then(function (response) {
      let tableSelector = d3.select("#historical-table");
      const predictionJson1 = response
      //
      let columns = [
        { head: 'Main Team', cl: 'table-head'},
        { head: 'Actual Games Won', cl: 'table-head'},
        { head: 'ML Predicted Wins', cl: 'table-head'},
        { head: 'Positive or Negative', cl: 'table-head'}
      ];
      //
      tableSelector.html("")
      let table = tableSelector.append("table")
      table.attr('class', "table table-striped table-bordered table-condensed").attr("id", "historical-table-id")
      table.append('thead').append('tr')
      .selectAll('th')
      .data(columns).enter()
      .append('th')
      .attr('class', function (d) {
        return d.cl;
      })
      .text(function (d) {
        return d.head;
      });

      let tbody = table.append("tbody")
      //Iterate JSON
      predictionJson1.forEach(mt =>{
        const row = tbody.append('tr')
        row.append("td").html( getTeamLogo(mt.team) + " " + getTeamName(mt.team) )
        row.append("td").html(mt.team_win_count)
        row.append("td").html(mt.predicted_win_count)
        if(mt.positive_or_negative == "+") {
          row.append("td").html( "<span class=\"badge badge-pill badge-success\"><i class=\"fa fa-check-circle\" aria-hidden=\"true\"></i></span>" )
        } else {
          row.append("td").html( "<i class=\"fa fa-minus\" aria-hidden=\"true\"></i>" )
        }
      });//End Main Team Loop
      //Conclude the table

  });
}


function init() {
  // Grab a reference to the dropdown select element
  const seasonSelector = d3.select("#selSeason");
  const modelSelector = d3.select("#selModel");
  //
  const modelCurrentSelector = d3.select("#selCurrentModel");
  //Use the list of sample names to populate the select options

  d3.json("/teamstat/metadata").then(function (response) {
    seasons = response.seasons;
    models = response.models;
    //
    seasons.forEach((season) => {
      seasonSelector
        .append("option")
        .text(season)
        .property("value", season);
    });
    //
    for (const [k, v] of Object.entries(models)) {
        modelSelector
        .append("option")
        .text(v)
        .property("value", k);
    }
    //
    for (const [k, v] of Object.entries(models)) {
      modelCurrentSelector
      .append("option")
      .text(v)
      .property("value", k);
    }

    const seasonCriteria = seasons[0]
    const modelCriteria = Object.keys(models)[0];
    const modelCurrentCriteria = Object.keys(models)[0];
    //Create the Chart
    buildCurrentSeasonPrediction(modelCurrentCriteria);
    buildPrediction(seasonCriteria, modelCriteria);
  });

}

function optionChangedCurrentModel(model) {
  // Fetch new data each time a new sample is selected
  let tableSelector = d3.select("#predictor-table");
  tableSelector.html("")
  tableSelector.html("<i class=\"fas fa-sync fa-spin\"></i>")
  buildCurrentSeasonPrediction(model);
}
function optionChangedSeason(season) {
  // Fetch new data each time a new sample is selected
  let tableSelector = d3.select("#historical-table");
  tableSelector.html("")
  tableSelector.html("<i class=\"fas fa-sync fa-spin\"></i>")
  buildPrediction(season, d3.select("#selModel").property("value"));
}
function optionChangedModel(model) {
  // Fetch new data each time a new sample is selected
  let tableSelector = d3.select("#historical-table");
  tableSelector.html("")
  tableSelector.html("<i class=\"fas fa-sync fa-spin\"></i>")
  buildPrediction(d3.select("#selSeason").property("value"), model);
}

init();