from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder

# Omniparser import 
from Omniparser_Usage.api import process_image


ActionType = Literal[
    "click", "double_click", "right_click",
    "hover", "type", "key", "scroll", "wait"
    ]
  
def ts() -> str:
    """Formatted timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class BrowserAgent:
    def __init__(self, driver: Optional[webdriver.Chrome] = None, *, config: Optional[Dict[str, Any]] = None, ) -> None:
        self.driver = driver
        # configuration
        default_config: Dict[str, Any] = {
            "screenshot_dir": "./screenshots",
            "wait_timeout": 15,
            "log_level": "INFO",
        }
        self.config = default_config | (config or {})

        # runtime context
        self.context: Dict[str, Any] = {
            "previous_actions": [],
            "session_id": ts(),
        }

        self._ensure_dirs()
        self._setup_logging()

        self.wait = WebDriverWait(self.driver, self.config["wait_timeout"])

        logging.info("BrowserAgent initialised.")


    # setup
    def _ensure_dirs(self) -> None:
        Path(self.config["screenshot_dir"]).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_path = Path(f"browser_agent_{self.context['session_id']}.log")
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"].upper(), "INFO"),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        )
    
    # Open URL
    
    def open(self, url: str):
        if self.driver is None:
            opts = Options()
            #opts.add_argument("--headless=new")
            #opts.add_argument("--window-size=1280,1000")
            self.driver = webdriver.Chrome(options=opts)  

        self.driver.get(url) 
        self.driver.maximize_window()
        time.sleep(2)  # quick’n’dirty wait; replace with WebDriverWait
                    
          
    def _ensure_driver(self):
        if self.driver is None:
            opts = Options()
            if self._headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--window-size=1280,1000")
            self.driver = webdriver.Chrome(options=opts)

    # Take screenshot 
    def take_screenshot(self) -> Path:
        if self.driver is None:
            self.driver = webdriver.Chrome
            print("Auto-created Chrome driver!")
        fname = (
            Path(self.config["screenshot_dir"])
            / f"screen_{self.context['session_id']}_{ts()}.png"
        )
        self.driver.save_screenshot(fname)
        logging.info("Screenshot saved → %s", fname)
        print(f"Screenshot saved → {fname}")
        return fname

    def _run_omniparser(self, img_path: Path) -> Dict[str, Any]:
        try:
            result = process_image(str(img_path))
            logging.info("OmniParser returned %s keys", len(result))
            os.makedirs("parser", exist_ok=True)

           
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"parser/parsed_output_{timestamp}.json")

            # Save result as JSON
            with output_path.open("w") as f:
                json.dump(result, f, indent=2)
            logging.info("OmniParser output saved to %s", output_path)
            return result
        except Exception as exc:
            logging.error("OmniParser failed: %s", exc)
            return {}
    
    def refresh_or_go_back(self, request: str) -> None:
        """Refresh the page or go back to the previous page."""
        if request == "refresh":
            self.driver.refresh()
            logging.info("Page refreshed.")
        elif request == "back":
            self.driver.back()
            logging.info("Navigated back to the previous page.")
        time.sleep(5)
    
    def _ensure_in_viewport(self, x: int, y: int) -> None:
        vw, vh = self.driver.execute_script(
            "return [window.innerWidth, window.innerHeight];"
        )
        if not (0 <= x <= vw and 0 <= y <= vh):
            raise ValueError(f"Target ({x},{y}) outside viewport {vw}×{vh}")

    def _click_at(self, x: int, y: int) -> None:
        """Absolute single click using W3C actions."""
        self._ensure_in_viewport(x, y)
        actions = ActionBuilder(                       # W3C-level builder
            self.driver,
            mouse=PointerInput("mouse", "mouse"),
        )
        actions.pointer_action.move_to_location(x, y)
        actions.pointer_action.click()
        actions.perform()

    def _move_to(self, x: int, y: int) -> None:
        """Move pointer to absolute (x,y) without clicking (uses ActionChains)."""
        self._ensure_in_viewport(x, y)
        body = self.driver.find_element(By.TAG_NAME, "body")
        ActionChains(self.driver) \
            .move_to_element_with_offset(body, 0, 0) \
            .move_by_offset(x, y) \
            .perform()

    # -------------------------------------------------------------------------
    # Helper to convert bbox → pixel centre (kept unchanged)
    # -------------------------------------------------------------------------
    def _viewport_point(self, bbox) -> tuple[int, int]:
        """
        `bbox` is [x1, y1, x2, y2] in **normalised** (0-1) coordinates.
        Returns pixel (x,y) of the rectangle centre.
        """
        vw, vh = self.driver.execute_script(
            "return [window.innerWidth, window.innerHeight];"
        )
        x1, y1, x2, y2 = bbox
        cx = int(((x1 + x2) / 2) * vw)
        cy = int(((y1 + y2) / 2) * vh)
        return cx, cy

    # -------------------------------------------------------------------------
    # MAIN EXECUTION ENTRY-POINT (only action logic changed)
    # -------------------------------------------------------------------------
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a low-level JSON action, performs it, and returns a result dict.
        """
        kind: ActionType = action["action"]
        res = {"success": True, "action": action}

        try:
            # ----------------------------------------------------------
            # Common coordinate preparation
            # ----------------------------------------------------------
            if "bbox" in action:
                vx, vy = self._viewport_point(action["bbox"])  # pixel centre

            # ----------------------------------------------------------
            # Pointer-based actions
            # ----------------------------------------------------------
            if kind == "click":
                self._click_at(vx, vy)
                time.sleep(5)
            elif kind == "type":
                self._click_at(vx, vy)                      # focus first
                active = self.driver.switch_to.active_element
                active.clear()
                active.send_keys(action["text"])

            # ----------------------------------------------------------
            # Scroll 
            # ----------------------------------------------------------
            elif kind == "scroll":
                dx = int(action.get("dx", 0))
                dy = int(action.get("dy", 0))
                self.driver.execute_script("window.scrollBy(arguments[0],arguments[1]);", dx, dy)

            elif kind == "double_click":
                body = self.driver.find_element(By.TAG_NAME, "body")
                ActionChains(self.driver) \
                    .move_to_element_with_offset(body, 0, 0) \
                    .move_by_offset(vx, vy) \
                    .double_click() \
                    .perform()

            elif kind == "right_click":
                body = self.driver.find_element(By.TAG_NAME, "body")
                ActionChains(self.driver) \
                    .move_to_element_with_offset(body, 0, 0) \
                    .move_by_offset(vx, vy) \
                    .context_click() \
                    .perform()

            elif kind == "hover":
                self._move_to(vx, vy)

            # ----------------------------------------------------------
            # Typing
            # ----------------------------------------------------------


            elif kind == "key":
                self.driver.switch_to.active_element.send_keys(action["key"])

            elif kind == "wait":
                time.sleep(action.get("seconds", 1))

            else:
                raise ValueError(f"Unknown action: {kind}")

        # --------------------------------------------------------------
        # Error handling
        # --------------------------------------------------------------
        except Exception as exc:
            logging.error("Action failed: %s", exc, exc_info=True)
            res.update(success=False, error=str(exc))

        

"""
    #Helper function 
    #Convert normalised bbox -> viewport pixel coords (centre). Scrolls the page to ensure the point is in view.
    def _viewport_point(self, bbox: List[float]) -> tuple[int, int]:

        x0, y0, x1, y1 = bbox
        vw, vh = self.driver.execute_script("return [window.innerWidth, window.innerHeight];")
        cx, cy = int((x0 + x1) / 2 * vw), int((y0 + y1) / 2 * vh)

        # if point is outside viewport, scroll so it lands ~middle
        if cy < 0 or cy > vh:
            self.driver.execute_script(
                "window.scrollTo(0, arguments[0] - window.innerHeight/2);",
                cy
            )
            cy = int(vh / 2)

        return cx, cy
    

    def _click_at(self, vx: int, vy: int):
        actions = ActionBuilder(
            self.driver,
            mouse=PointerInput(PointerInput.INTERACTION_MOUSE, "mouse")
        )
        actions.pointer_action.move_to_location(vx, vy)
        actions.pointer_action.click()
        actions.perform()


    # execute_action
    # ActionType = Literal[
    # "click", "double_click", "right_click",
    # "hover", "type", "key", "scroll", "wait"
    # ]
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        kind: ActionType = action["action"]       
        res = {"success": True, "action": action}
        try:
            # coordinate helpers 
            if "bbox" in action:
                vx, vy = self._viewport_point(action["bbox"])  # centre of box

            # actions 
            if kind == "click":
                #ActionChains(self.driver).move_by_offset(vx, vy).click().perform()
                #self.driver.execute_script("window.scrollBy(0,0)")
                self._click_at(vx, vy)

            elif kind == "double_click":
                ActionChains(self.driver).move_by_offset(vx, vy).double_click().perform()
                self.driver.execute_script("window.scrollBy(0,0)") 

            elif kind == "right_click":
                ActionChains(self.driver).move_by_offset(vx, vy).context_click().perform()
                self.driver.execute_script("window.scrollBy(0,0)")

            elif kind == "hover":
                ActionChains(self.driver).move_by_offset(vx, vy).perform()
                self.driver.execute_script("window.scrollBy(0,0)")

            # typing  
            elif kind == "type":
                ActionChains(self.driver).move_by_offset(vx, vy).click().perform()                    # focus first
                self.driver.switch_to.active_element.clear()
                self.driver.switch_to.active_element.send_keys(action["text"])

            #  scroll
            elif kind == "scroll":
                dx = int(action.get("dx", 0))
                dy = int(action.get("dy", 0))
                self.driver.execute_script("window.scrollBy(arguments[0],arguments[1]);", dx, dy)

            #  wait
            elif kind == "wait":
                time.sleep(action.get("seconds", 1))

            else:
                raise ValueError(f"Unknown action: {kind}")

        except Exception as exc:                     # broad catch inside agent
            logging.error("Action failed: %s", exc, exc_info=True)
            res.update(success=False, error=str(exc))


"""
    
