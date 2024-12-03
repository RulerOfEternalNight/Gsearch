$(document).ready(function(){
    $(".golist").click(function(){
        $(".file-field").addClass("listview");
    });
    $(".gogrid").click(function(){
        $(".file-field").removeClass("listview");
    });
});