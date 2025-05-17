async function run() {
  const query = document.getElementById("query").value;

  const response = await fetch("/run", {
      method: "POST",
      headers: {
          "Content-Type": "application/json"
      },
      body: JSON.stringify({ input_data: query })
  });

  const data = await response.json();
  document.getElementById("result").textContent = "Result: " + data.result;
}
