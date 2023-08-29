/*
Example table

| base / target       | YOLOv5 | YOLOv7 | [YOLOv8](https://github.com/autodistill/autodistill-yolov8) | RT-DETR |
|:-------------------:|:------:|:------:|:------:|:-------:|
| GroundingDINO       |        |        |        |         |
| [Grounded SAM](https://github.com/autodistill/autodistill-grounded-sam)        |        |        | ✅     |         |
| DETIC               |        |        |        |         |
| OWL-ViT             |        |        |        |         |
*/

const fs = require("fs");
const path = require("path");
const _ = require("lodash");

// pull in README.md
var readme = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");

// read models.csv
const modelFile = fs.readFileSync(path.join(__dirname, "models.csv"), "utf8");
const lines = modelFile.split("\n");
const header = lines[0].split(",").map((column) => column.trim());

const rows = lines.slice(1).filter((row) => row && row.length && row.includes(",") && !row.startsWith("#"));;
const models = _.map(rows, function(row) {
    var values = row.split(",").map((column) => column.trim());
    return _.zipObject(header, values);
});

_.each(["object-detection", "instance-segmentation", "classification"], function(task) {
    var base_models = _.filter(models, function(model) {
        return model["task"] == task && model["is_base_model"] == "1";
    });

    var target_models = _.filter(models, function(model) {
        return model["task"] == task && model["is_base_model"] == "0";
    });

    // construct table string for task
    var table = "| base / target | " + _.map(target_models, function(model) {
        if(model["repo_url"]) {
            return "[" + model["display_name"] + "](https://github.com/" + model["repo_url"] + ")";
        } else {
            return model["display_name"];
        }
    }).join(" | ") + " |\n";

    table += "|:---:|" + _.map(target_models, function(model) {
        return ":---:|";
    }).join("");

    table += "\n";

    _.each(base_models, function(base_model) {
        var display_name = base_model["display_name"];
        if(base_model["repo_url"]) display_name = "[" + base_model["display_name"] + "](https://github.com/" + base_model["repo_url"] + ")";
        table += "| " + display_name + " | " + _.map(target_models, function(target_model) {
            var status = "";

            // if both completed, checkmark emoji
            // if both in progress or one in progress and one completed, construction emoji
            if(base_model["status"] == "completed" && target_model["status"] == "completed") {
                status = "✅";
            } else if(base_model["status"] == "completed" && target_model["status"] == "in-progress") {
                status = "🚧";
            } else if(base_model["status"] == "in-progress" && target_model["status"] == "completed") {
                status = "🚧";
            } else if(base_model["status"] == "in-progress" && target_model["status"] == "in-progress") {
                status = "🚧";
            }

            return status;
        }).join(" | ") + " |\n";
    });

    // console.log(table);
    // console.log("\n\n");

    // replace table in README.md
    // should the lines between `### {task}` and the start of the next section (`##`)
    var taskHumanReadable = task.replace("-", " ");
    var taskHeader = "### " + taskHumanReadable;
    var nextHeader = "##";
    var taskHeaderIndex = readme.indexOf(taskHeader);
    var nextHeaderIndex = readme.indexOf(nextHeader, taskHeaderIndex + taskHeader.length);
    var taskSection = readme.substring(taskHeaderIndex, nextHeaderIndex);
    readme = readme.replace(taskSection, taskHeader + "\n\n" + table + "\n\n");
});

console.log(readme);
// overwrite file
fs.writeFileSync(path.join(__dirname, "..", "README.md"), readme, "utf8");
// save models.csv to ../autodistill
fs.writeFileSync(path.join(__dirname, "..", "..", "autodistill", "models.csv"), modelFile, "utf8");