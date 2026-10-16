import { feature_names } from './features_names.js'
import { SparseAutoencoder } from './sparse_autoencoder.js'


function debounce(func, delay = 500) {
    let timeoutId = null;
    let lastArgs = null;
    let lastContext = null;

    const executeTrailing = () => {
        if (lastArgs) {
            func.apply(lastContext, lastArgs);
            lastArgs = null;
            lastContext = null;
            timeoutId = setTimeout(executeTrailing, delay);
        } else {
            timeoutId = null;
        }
    };

    return function (...args) {
        if (!timeoutId) {
            func.apply(this, args);
            timeoutId = setTimeout(executeTrailing, delay);
        } else {
            lastArgs = args;
            lastContext = this;
        }
    };
}


class Slider {
	constructor(index, debouncedReEval) {
		this.index = index;
		this.debouncedReEval = debouncedReEval;
		this.enabled = false;
		this.element = document.createElement("slider-container");
		this.checkbox = document.createElement("input");
		this.checkbox.type = "checkbox";
		this.checkbox.checked = false;
		this.checkbox.addEventListener("change", () => this.updateCheckbox());
		this.element.appendChild(this.checkbox);
		const p1 = document.createElement("label");
		p1.innerHTML = feature_names[index][0];
		this.element.appendChild(p1);
		this.range = document.createElement("input");
		this.range.type = "range";
		this.range.min = -1;
		this.range.max = 1;
		this.range.step = 0.02;
		this.range.disabled = !this.enabled;
		this.range.value = 0;
		this.range.addEventListener("input", () => this.updateSlider());
		this.element.appendChild(this.range);
		this.element.style.setProperty("--color1", feature_names[index][2]);
		this.element.style.setProperty("--color2", feature_names[index][3]);
		this.updateColor();
		const p2 = document.createElement("label");
		p2.innerHTML = feature_names[index][1];
		this.element.appendChild(p2);
	}
	updateCheckbox() {
		if (this.checkbox.checked != this.enabled) {
			this.enabled = this.checkbox.checked;
			this.range.disabled = !this.enabled;
			this.debouncedReEval();
		}
		this.updateColor();
	}
	updateSlider() {
		if (this.enabled) {
			this.debouncedReEval();
		}
		this.updateColor();
	}
	updateColor() {
		this.element.style.setProperty("--user-selected", this.enabled ? "100%" : "0%");
		this.element.style.setProperty("--color-mix", (Number(this.range.value) + 1.) * 50. + "%");
	}
	getValue() {
		if (this.enabled) {
			return this.range.value;
		} else {
			return NaN;
		}
	}
	setValue(val) {
		if (!this.enabled) {
			this.range.value = val / 0.85; // this makes it just a bit more interesting
			this.updateColor();
		}
	}
}

function reEval(model, sliders) {
	// get current values
	const current = new Float32Array(500);
	for (let i=0; i<500; i++) {
		current[i] = sliders[i].getValue();
	}
	
	// run model
	const result = model.full(current);

	// set the new values
	for (let i=0; i<500; i++) {
		sliders[i].setValue(result[i]);
	}

}

function main() {
	const sliders = [];

	SparseAutoencoder.load("model.bin").
	catch(err => console.log(err))
	.then(val => {
		LOADING_MESSAGE.remove();
		const model = val;

		const debouncedReEval = debounce(() => reEval(model, sliders), 100);

		for (let i=0; i<500; i++) {
			const slider = new Slider(i, debouncedReEval);
			sliders.push(slider);
			CONTENT_MAIN.appendChild(slider.element);
		}
		reEval(model, sliders);
		
	});
}

main();