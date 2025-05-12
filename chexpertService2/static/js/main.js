$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html(data);
            },
            error: function () {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html("<div class='alert alert-danger'>Prediction failed. Please try again.</div>");
            }
        });
    });
});

// Feedback form handler
$(document).on('submit', '#feedback-form', function (e) {
    e.preventDefault();

    // Build label string (from 14 dropdowns)
    let labels = [];
    for (let i = 0; i < 14; i++) {
        const val = $(`select[name='label_${i}']`).val();
        labels.push(val || ""); // push empty string if none selected
    }

    // Set the final labels string into a hidden input (or add dynamically)
    if ($('#feedback-form input[name="labels"]').length === 0) {
        $('<input>').attr({
            type: 'hidden',
            name: 'labels',
            value: labels.join(',')
        }).appendTo('#feedback-form');
    } else {
        $('#feedback-form input[name="labels"]').val(labels.join(','));
    }

    // Submit form with AJAX
    const formData = new FormData(this);
    $.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:8000/submit-feedback',  // or use your actual FastAPI host
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            alert(data.message);
        },
        error: function (xhr) {
            alert("Feedback failed: " + xhr.responseJSON.detail);
        }
    });
});
