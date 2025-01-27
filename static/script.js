document.addEventListener("DOMContentLoaded", () => {
    const editButton = document.getElementById("editButton");
    const saveButton = document.getElementById("saveButton");
    const inputs = document.querySelectorAll(".member-item input");

    editButton.addEventListener("click", () => {
        inputs.forEach(input => input.removeAttribute("readonly"));
        editButton.style.display = "none";
        saveButton.style.display = "block";
    });
});
