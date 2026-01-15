const readJsonData = (id) => {
  const element = document.getElementById(id);
  if (!element) {
    return null;
  }
  try {
    return JSON.parse(element.textContent);
  } catch (error) {
    return null;
  }
};

const forecastData = readJsonData("forecast-chart-data");
const forecastCanvas = document.getElementById("forecastChart");

if (forecastCanvas && forecastData && forecastData.labels.length) {
  const ctx = forecastCanvas.getContext("2d");
  const datasets = [
    {
      label: "Prediction",
      data: forecastData.prediction,
      borderColor: "#4f46e5",
      backgroundColor: "rgba(79, 70, 229, 0.2)",
      fill: true,
      tension: 0.35,
    },
  ];

  const hasIntervals = forecastData.lower_bound.some((value) => value !== null);
  if (hasIntervals) {
    datasets.push(
      {
        label: "Lower bound",
        data: forecastData.lower_bound,
        borderColor: "rgba(148, 163, 184, 0.9)",
        borderDash: [6, 6],
        fill: false,
        tension: 0.2,
      },
      {
        label: "Upper bound",
        data: forecastData.upper_bound,
        borderColor: "rgba(148, 163, 184, 0.9)",
        borderDash: [6, 6],
        fill: false,
        tension: 0.2,
      }
    );
  }

  new Chart(ctx, {
    type: "line",
    data: {
      labels: forecastData.labels,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
        },
      },
      scales: {
        y: {
          grid: {
            color: "rgba(148, 163, 184, 0.2)",
          },
        },
      },
    },
  });
}

const jobStatusData = readJsonData("job-status-chart-data");
const jobCanvas = document.getElementById("jobStatusChart");

if (jobCanvas && jobStatusData) {
  const ctx = jobCanvas.getContext("2d");
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: jobStatusData.labels,
      datasets: [
        {
          data: jobStatusData.values,
          backgroundColor: ["#f59e0b", "#0ea5e9", "#22c55e", "#ef4444"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

const parsePayload = (form) => {
  const payload = {};
  const formData = new FormData(form);
  form.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
    payload[checkbox.name] = checkbox.checked;
  });
  for (const [key, value] of formData.entries()) {
    if (key === "csrfmiddlewaretoken") {
      continue;
    }
    if (value !== "" && !(key in payload)) {
      payload[key] = value;
    }
  }
  return payload;
};

document.querySelectorAll(".action-form").forEach((form) => {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const endpoint = form.dataset.endpoint;
    const output = form.querySelector(".action-output");
    const csrfToken = form.querySelector("input[name='csrfmiddlewaretoken']")?.value;
    const payload = parsePayload(form);

    output.classList.remove("error", "success");
    output.textContent = "Submitting request...";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfToken || "",
        },
        body: JSON.stringify(payload),
      });
      const text = await response.text();
      if (text) {
        try {
          const parsed = JSON.parse(text);
          output.textContent = JSON.stringify(parsed, null, 2);
        } catch (parseError) {
          output.textContent = text;
        }
      } else {
        output.textContent = response.ok ? "Done." : "Request failed.";
      }
      output.classList.add(response.ok ? "success" : "error");
    } catch (error) {
      output.textContent = `Error: ${error.message}`;
      output.classList.add("error");
    }
  });
});
