function sendData() {
    const inputText = document.getElementById("inputText").value;
    const url = "http://127.0.0.1:5000/test";

    fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_input: inputText }),
    })
        .then((response) => response.json())
        .then((data) => {
            document.getElementById("result").innerHTML =
                "Server Response: " + data.result;
        })
        .catch((error) => {
            console.error("Error:", error);
            document.getElementById("result").innerHTML =
                "Error occurred. Check console for details.";
        });
}
