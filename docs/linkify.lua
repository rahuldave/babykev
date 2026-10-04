-- Cross-references: a backticked name in prose becomes a link to its reference entry,
-- when the name is a documented symbol. The index is written by apidocs.py.
local symbols = {}
local f = io.open(quarto.project.directory .. "/_symbols.json", "r")
if f then
  symbols = quarto.json.decode(f:read("a"))
  f:close()
end

return {
  {
    traverse = "topdown",
    Link = function(el) return el, false end,   -- leave existing links alone
    Code = function(el)
      local target = symbols[(el.text:gsub("%(%)$", ""))]
      if target then return pandoc.Link(el, target), false end   -- false: do not revisit the new link
    end,
  },
}
