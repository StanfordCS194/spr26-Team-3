/**
 * Vanilla three.js mesh viewer. No drei / no react-three-fiber so it works
 * cleanly under React 19. Loads a .ply and frames the camera to it (the mesh
 * is kept in real coordinates so an editable placement transform — rotate /
 * move / scale, applied about the mesh centre — previews exactly what gets
 * baked into the file). Gives the user OrbitControls + a grid floor + lights.
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import { cn } from "@/lib/utils";

export type Placement = {
  rx: number; // rotation degrees about X
  ry: number;
  rz: number;
  tx: number; // translation (world units)
  ty: number;
  tz: number;
  scale: number; // uniform scale (>0)
};

export const IDENTITY_PLACEMENT: Placement = {
  rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0, scale: 1,
};

type LoadState = { state: "loading" | "ready" | "error"; msg?: string };

/** Placement matrix that rotates & scales about `center`, then translates. */
function placementMatrix(p: Placement, center: THREE.Vector3): THREE.Matrix4 {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(p.rx),
    THREE.MathUtils.degToRad(p.ry),
    THREE.MathUtils.degToRad(p.rz),
    "XYZ",
  );
  const R = new THREE.Matrix4().makeRotationFromEuler(euler);
  const S = new THREE.Matrix4().makeScale(p.scale, p.scale, p.scale);
  const toOrigin = new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z);
  const back = new THREE.Matrix4().makeTranslation(center.x, center.y, center.z);
  const T = new THREE.Matrix4().makeTranslation(p.tx, p.ty, p.tz);
  // M = T · back · R · S · toOrigin  (apply rightmost first)
  return new THREE.Matrix4()
    .multiply(T)
    .multiply(back)
    .multiply(R)
    .multiply(S)
    .multiply(toOrigin);
}

export function MeshViewer({
  url,
  className,
  placement = IDENTITY_PLACEMENT,
  onMatrix,
}: {
  url: string;
  className?: string;
  placement?: Placement;
  onMatrix?: (elements: number[]) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [load, setLoad] = useState<LoadState>({ state: "loading" });
  // Live handles shared between the load effect and the placement effect.
  const envRef = useRef<{ group: THREE.Group; center: THREE.Vector3 } | null>(null);
  const placementRef = useRef(placement);
  placementRef.current = placement;
  const applyPlacementRef = useRef<(() => void) | null>(null);
  const onMatrixRef = useRef(onMatrix);
  onMatrixRef.current = onMatrix;

  // Scene setup + mesh load — re-runs only when the URL changes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !url) return;
    setLoad({ state: "loading" });
    envRef.current = null;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 2000);
    camera.position.set(4, 3, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.95);
    sun.position.set(8, 12, 6);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.25);
    fill.position.set(-6, 4, -5);
    scene.add(fill);

    const grid = new THREE.GridHelper(40, 40, 0x2a2a2a, 0x161616);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.7;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const envGroup = new THREE.Group();
    envGroup.matrixAutoUpdate = false;
    scene.add(envGroup);

    const loader = new PLYLoader();
    loader.load(
      url,
      (geom) => {
        if (!geom.getAttribute("position")?.count) {
          setLoad({ state: "error", msg: "The 3D scene came out empty. Try a clip with more overlap and detail." });
          return;
        }
        geom.computeVertexNormals();
        geom.computeBoundingBox();
        geom.computeBoundingSphere();
        const center = geom.boundingBox!.getCenter(new THREE.Vector3());
        const radius = geom.boundingSphere?.radius || 1;

        const material = new THREE.MeshStandardMaterial({
          color: 0xdcdcdc,
          roughness: 0.55,
          metalness: 0.05,
          side: THREE.DoubleSide,
          vertexColors: !!geom.getAttribute("color"),
        });
        const mesh = new THREE.Mesh(geom, material);
        envGroup.add(mesh);
        envRef.current = { group: envGroup, center };

        // Frame the camera to the mesh and orbit around its centre.
        const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2)) * 1.3;
        camera.position.copy(center).add(new THREE.Vector3(1, 0.7, 1).normalize().multiplyScalar(dist));
        camera.far = Math.max(2000, dist * 50);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();

        applyPlacement();
        setLoad({ state: "ready" });
      },
      undefined,
      (err) => {
        console.error("PLY load failed", err);
        setLoad({ state: "error", msg: "Couldn't load the 3D scene — it may be missing or failed to generate." });
      },
    );

    // Applies the latest placement prop to the loaded group + reports the matrix.
    function applyPlacement() {
      const env = envRef.current;
      if (!env) return;
      const M = placementMatrix(placementRef.current, env.center);
      env.group.matrix.copy(M);
      env.group.matrixWorldNeedsUpdate = true;
      onMatrixRef.current?.(M.elements.slice());
    }
    applyPlacementRef.current = applyPlacement;

    let rafId = 0;
    const tick = () => {
      rafId = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      envRef.current = null;
      applyPlacementRef.current = null;
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Re-apply placement live when it changes (no mesh reload).
  useEffect(() => {
    applyPlacementRef.current?.();
  }, [placement]);

  return (
    <div
      className={cn(
        "relative w-full h-full border border-border rounded-sm overflow-hidden bg-[oklch(0.06_0_0)]",
        className,
      )}
    >
      <div ref={containerRef} className="absolute inset-0" />
      {load.state !== "ready" && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 text-center">
          {load.state === "loading" ? (
            <span className="mono text-xs text-muted-foreground animate-pulse">
              Loading 3D scene…
            </span>
          ) : (
            <span className="mono text-xs text-[var(--status-fail)] max-w-xs">
              {load.msg ?? "Couldn't load the 3D scene."}
            </span>
          )}
        </div>
      )}
      <div className="absolute pointer-events-none top-3 right-3 mono text-[10px] text-muted-foreground/70 px-2 py-1 rounded-sm bg-background/60 backdrop-blur-sm">
        drag <span className="text-foreground">orbit</span> · shift+drag{" "}
        <span className="text-foreground">pan</span> · wheel{" "}
        <span className="text-foreground">zoom</span>
      </div>
    </div>
  );
}
