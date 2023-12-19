(function (H) {
    H.seriesTypes.pie.prototype.animate = function (init) {
        const series = this,
            chart = series.chart,
            points = series.points,
            {
                animation
            } = series.options,
            {
                startAngleRad
            } = series;

        function fanAnimate(point, startAngleRad) {
            const graphic = point.graphic,
                args = point.shapeArgs;

            if (graphic && args) {

                graphic
                    // Set inital animation values
                    .attr({
                        start: startAngleRad,
                        end: startAngleRad,
                        opacity: 1
                    })
                    // Animate to the final position
                    .animate({
                        start: args.start,
                        end: args.end
                    }, {
                        duration: animation.duration / points.length
                    }, function () {
                        // On complete, start animating the next point
                        if (points[point.index + 1]) {
                            fanAnimate(points[point.index + 1], args.end);
                        }
                        // On the last point, fade in the data labels, then
                        // apply the inner size
                        if (point.index === series.points.length - 1) {
                            series.dataLabelsGroup.animate({
                                opacity: 1
                            },
                            void 0,
                            function () {
                                points.forEach(point => {
                                    point.opacity = 1;
                                });
                                series.update({
                                    enableMouseTracking: true
                                }, false);
                                chart.update({
                                    plotOptions: {
                                        pie: {
                                            innerSize: '40%',
                                            borderRadius: 8
                                        }
                                    }
                                });
                            });
                        }
                    });
            }
        }

        if (init) {
            // Hide points on init
            points.forEach(point => {
                point.opacity = 0;
            });
        } else {
            fanAnimate(points[0], startAngleRad);
        }
    };
}(Highcharts));

Highcharts.chart('skin_container1', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.16
        }, {
            name: '민감성',
            y: 0.16
        }, {
            name: '복합성',
            y: 0.5
        }, {
            name: '중성',
            y: 0.08
        }, {
            name: '지성',
            y: 0.1
        }]
    }]
});

Highcharts.chart('skin_container2', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.18
        }, {
            name: '민감성',
            y: 0.1
        }, {
            name: '복합성',
            y: 0.54
        }, {
            name: '중성',
            y: 0.04
        }, {
            name: '지성',
            y: 0.14
        }]
    }]
});

Highcharts.chart('skin_container3', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.16
        }, {
            name: '민감성',
            y: 0.18
        }, {
            name: '복합성',
            y: 0.48
        }, {
            name: '중성',
            y: 0.06
        }, {
            name: '지성',
            y: 0.12
        }]
    }]
});

Highcharts.chart('skin_container4', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.26
        }, {
            name: '민감성',
            y: 0.22
        }, {
            name: '복합성',
            y: 0.34
        }, {
            name: '중성',
            y: 0.08
        }, {
            name: '지성',
            y: 0.1
        }]
    }]
});

Highcharts.chart('skin_container5', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.18
        }, {
            name: '민감성',
            y: 0.1
        }, {
            name: '복합성',
            y: 0.56
        }, {
            name: '중성',
            y: 0
        }, {
            name: '지성',
            y: 0.16
        }]
    }]
});

Highcharts.chart('skin_container6', {
    chart: {
        type: 'pie'
    },
    title: {
        text: '피부 타입',
        align: 'center'
    },
    subtitle: {
        text: '사용자의 피부 타입 분포도',
        align: 'center'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            borderWidth: 2,
            cursor: 'pointer',
            dataLabels: {
                enabled: true,
                format: '<b>{point.name}</b><br>{point.percentage}%',
                distance: 20
            }
        }
    },
    series: [{
        // Disable mouse tracking on load, enable after custom animation
        enableMouseTracking: false,
        animation: {
            duration: 2000
        },
        colorByPoint: true,
        data: [{
            name: '건성',
            y: 0.3
        }, {
            name: '민감성',
            y: 0.16
        }, {
            name: '복합성',
            y: 0.44
        }, {
            name: '중성',
            y: 0
        }, {
            name: '지성',
            y: 0.1
        }]
    }]
});
